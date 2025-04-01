# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import logging
import torch
import time
from peft import PeftModel
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh
from collections import OrderedDict

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.utils.debug import log_gpu_memory_usage

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def diagnose_embedding_weights(model, model_type="Unknown", prefix="", rank=0, tp_rank=-1):
    """
    Utility function to diagnose embedding weights in different types of models.
    
    Args:
        model: Model to examine
        model_type: String description of model type
        prefix: Prefix for logging
        rank: Process rank
        tp_rank: Tensor parallel rank
    """
    try:
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Diagnosing embeddings for {model_type} model type: {type(model).__name__}")
        
        # For PEFT models, we need to check the base model
        if isinstance(model, PeftModel):
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Model is a PeftModel, checking base_model.model")
            base_model = model.base_model.model
            
            # Check embedding layer
            if hasattr(base_model, 'embed_tokens'):
                emb = base_model.embed_tokens.weight
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found embed_tokens: shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
                return
            
            # Try model.embed_tokens
            if hasattr(base_model, 'model') and hasattr(base_model.model, 'embed_tokens'):
                emb = base_model.model.embed_tokens.weight
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found model.embed_tokens: shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
                return
        
        # For regular models, try common embedding patterns
        # Try embed_tokens directly
        if hasattr(model, 'embed_tokens'):
            emb = model.embed_tokens.weight
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found embed_tokens: shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
            return
        
        # Try model.embed_tokens
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            emb = model.model.embed_tokens.weight
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found model.embed_tokens: shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
            return
        
        # Try transformer.wte
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            emb = model.transformer.wte.weight
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found transformer.wte: shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
            return
            
        # Check state_dict as a last resort
        try:
            state_dict = model.state_dict()
            emb_keys = [k for k in state_dict.keys() if "embed_tokens.weight" in k or "wte.weight" in k or "word_embeddings.weight" in k]
            
            if emb_keys:
                for emb_key in emb_keys:
                    emb = state_dict[emb_key]
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Found in state_dict: {emb_key}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Shape={emb.shape}, mean={emb.mean().item():.6f}, std={emb.std().item():.6f}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Non-zero elements: {(emb != 0).sum().item()}, device={emb.device}")
                return
        except Exception as e:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Error accessing state_dict: {e}")
            
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Could not find embedding weights in model")
            
    except Exception as e:
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] {prefix} Error diagnosing embedding weights: {e}")


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        rank = torch.distributed.get_rank()
        tp_rank = vllm_ps.get_tensor_model_parallel_rank() if hasattr(vllm_ps, 'is_initialized') and vllm_ps.is_initialized() else -1
        tp_size = vllm_ps.get_tensor_model_parallel_world_size() if hasattr(vllm_ps, 'is_initialized') and vllm_ps.is_initialized() else 1
        print(f"\n[ShardingManager Rank {rank} TP {tp_rank}/{tp_size}] === ENTER ===")
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] FSDP Module type: {type(self.module)}")
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Wrapped Module type: {type(self.module._fsdp_wrapped_module)}")
        
        log_gpu_memory_usage(f'[ShardingManager Rank {rank} TP {tp_rank}] Before state_dict()', logger=logger)
        
        # === PRINT 5: Check source weights before merge/state_dict ===
        is_peft_model = isinstance(self.module._fsdp_wrapped_module, PeftModel)
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Is PeftModel: {is_peft_model}")
        
        # Use the diagnose utility to check wrapped module before merge
        with FSDP.summon_full_params(self.module, writeback=False, rank0_only=False):
            diagnose_embedding_weights(
                self.module._fsdp_wrapped_module, 
                model_type="FSDP Wrapped", 
                prefix="BEFORE MERGE",
                rank=rank, 
                tp_rank=tp_rank
            )
        
        if is_peft_model or True:  # Check even if not PEFT for baseline
            try:
                with FSDP.summon_full_params(self.module, writeback=False, rank0_only=False):
                    if is_peft_model:
                        source_module = self.module._fsdp_wrapped_module.base_model.model
                    else:
                        source_module = self.module._fsdp_wrapped_module
                    
                    # Try to find embedding weights
                    emb_weight = None
                    emb_key = None
                    for key in ["embed_tokens.weight", "model.embed_tokens.weight", "transformer.wte.weight", "word_embeddings.weight"]:
                        try:
                            if "." in key:
                                parts = key.split(".")
                                module_ref = source_module
                                for part in parts[:-1]:
                                    if hasattr(module_ref, part):
                                        module_ref = getattr(module_ref, part)
                                    else:
                                        break
                                if hasattr(module_ref, parts[-1]):
                                    emb_weight = getattr(module_ref, parts[-1])
                                    emb_key = key
                                    break
                            else:
                                if hasattr(source_module, key):
                                    emb_weight = getattr(source_module, key)
                                    emb_key = key
                                    break
                        except:
                            continue
                    
                    if emb_weight is not None:
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Source Embedding Weight ('{emb_key}') found pre-merge")
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Source Embedding Weight Shape pre-merge: {emb_weight.shape}")
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Source Embedding Weight Mean pre-merge: {emb_weight.mean().item():.6f}")
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Source Embedding Weight Std pre-merge: {emb_weight.std().item():.6f}")
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Source Embedding Weight Device pre-merge: {emb_weight.device}")
                    else:
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Could not find source embedding weights pre-merge/state_dict!")
            except Exception as e:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error checking source weights: {e}")
        
        torch.distributed.barrier()
        # ============================================================
        
        params = None  # Initialize params
        if is_peft_model:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Merging adapter...")
            with FSDP.summon_full_params(self.module):
                self.module._fsdp_wrapped_module.merge_adapter()
                
            # Diagnose after merge
            diagnose_embedding_weights(
                self.module._fsdp_wrapped_module, 
                model_type="PEFT Model (after merge)", 
                prefix="AFTER MERGE",
                rank=rank, 
                tp_rank=tp_rank
            )
                
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Adapter merged. Getting state_dict from base_model.model...")
            params = self.module._fsdp_wrapped_module.base_model.model.state_dict()
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Got state_dict. Num keys: {len(params)}")

            # === PRINT 6: Check LoRA state_dict keys ===
            keys = list(params.keys())
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: State dict keys (first 5): {keys[:5]}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: State dict keys (last 5): {keys[-5:]}")
            
            # Check for embedding weights in state dict
            emb_keys = [k for k in keys if "embed_tokens.weight" in k or "wte.weight" in k or "word_embeddings.weight" in k]
            for emb_key in emb_keys:
                emb_weight_val = params[emb_key]
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Found embedding key: {emb_key}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight shape in state_dict: {emb_weight_val.shape}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight mean in state_dict: {emb_weight_val.mean().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight std in state_dict: {emb_weight_val.std().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight device in state_dict: {emb_weight_val.device}")
            
            if not emb_keys:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: No embedding keys found in state_dict!")
            # ==========================================

            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Filtering LoRA params from state_dict...")
            # Save original params copy for comparison
            params_before = OrderedDict(params)
            
            # FIXME: use more rigorous way to filter out the adapter weights
            params = OrderedDict((k.replace(".base_layer.", "."), v) for k, v in params.items() if not ".lora_" in k)
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: State_dict filtered. Num keys: {len(params)}")

            # === PRINT 7: Check LoRA state_dict keys AFTER filter ===
            keys = list(params.keys())
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Filtered state dict keys (first 5): {keys[:5]}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Filtered state dict keys (last 5): {keys[-5:]}")
            
            # Check for embedding weights in filtered state dict
            emb_keys_post_filter = [k for k in keys if "embed_tokens.weight" in k or "wte.weight" in k or "word_embeddings.weight" in k]
            for emb_key in emb_keys_post_filter:
                emb_weight_val = params[emb_key]
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Found embedding key POST-FILTER: {emb_key}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight mean POST-FILTER: {emb_weight_val.mean().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight std POST-FILTER: {emb_weight_val.std().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding weight device POST-FILTER: {emb_weight_val.device}")
            
            if not emb_keys_post_filter:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Embedding keys MISSING POST-FILTER!")
            
            # If original keys were found but filtered keys are missing, we have a problem
            if emb_keys and not emb_keys_post_filter:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] !!!!! EMBEDDING KEYS WERE FILTERED OUT !!!!!")
                
                # Find what happened to the keys
                for emb_key in emb_keys:
                    if emb_key not in emb_keys_post_filter:
                        # Check if it was renamed
                        for new_key in keys:
                            if emb_key.replace(".base_layer.", ".") == new_key:
                                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Embedding key {emb_key} was renamed to {new_key}")
                        # If not found after rename, was it completely removed?
                        emb_key_renamed = emb_key.replace(".base_layer.", ".")
                        if emb_key_renamed not in keys:
                            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Embedding key {emb_key} (renamed: {emb_key_renamed}) was completely removed!")
                
                # Add embedding keys back to params
                for emb_key in emb_keys:
                    key_renamed = emb_key.replace(".base_layer.", ".")
                    params[key_renamed] = params_before[emb_key]
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] Added embedding key back: {key_renamed}")
            # =======================================================
        else:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Getting state_dict...")
            params = self.module.state_dict()
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Got state_dict. Num keys: {len(params)}")

            # === PRINT 8: Check Non-LoRA state_dict keys ===
            keys = list(params.keys())
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: State dict keys (first 5): {keys[:5]}")
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: State dict keys (last 5): {keys[-5:]}")
            
            # Check for embedding weights in state dict
            emb_keys = [k for k in keys if "embed_tokens.weight" in k or "wte.weight" in k or "word_embeddings.weight" in k]
            for emb_key in emb_keys:
                emb_weight_val = params[emb_key]
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Found embedding key: {emb_key}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Embedding weight shape: {emb_weight_val.shape}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Embedding weight mean: {emb_weight_val.mean().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Embedding weight std: {emb_weight_val.std().item():.6f}")
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: Embedding weight device: {emb_weight_val.device}")
            
            if not emb_keys:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Non-LoRA Path: No embedding keys found in state_dict!")
            # ============================================

        log_gpu_memory_usage(f'[ShardingManager Rank {rank} TP {tp_rank}] After state_dict()', logger=logger)
        
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Using load_format: {load_format}")
        
        # Diagnose vLLM model before sync
        try:
            diagnose_embedding_weights(
                self.inference_engine.llm_engine.model,
                model_type="vLLM Model",
                prefix="BEFORE SYNC",
                rank=rank,
                tp_rank=tp_rank
            )
        except Exception as e:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error diagnosing vLLM model before sync: {e}")
        
        # === PRINT 9: Check vLLM weights BEFORE sync ===
        try:
            if hasattr(self.inference_engine.llm_engine, 'model'):
                vllm_model = self.inference_engine.llm_engine.model
                # Try common embedding attribute patterns
                vllm_emb_weight = None
                emb_attr_name = None
                
                for attr_name in ["embed_tokens.weight", "model.embed_tokens.weight", "transformer.wte.weight", "word_embeddings.weight"]:
                    try:
                        parts = attr_name.split(".")
                        curr_module = vllm_model
                        for part in parts[:-1]:
                            if hasattr(curr_module, part):
                                curr_module = getattr(curr_module, part)
                            else:
                                break
                        if hasattr(curr_module, parts[-1]):
                            vllm_emb_weight = getattr(curr_module, parts[-1])
                            emb_attr_name = attr_name
                            break
                    except:
                        continue
                
                if vllm_emb_weight is not None:
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight ('{emb_attr_name}') found BEFORE sync")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Shape BEFORE sync: {vllm_emb_weight.shape}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Mean BEFORE sync: {vllm_emb_weight.mean().item():.6f}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Std BEFORE sync: {vllm_emb_weight.std().item():.6f}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Device BEFORE sync: {vllm_emb_weight.device}")
                else:
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] Could not find vLLM embedding weights BEFORE sync")
            else:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Cannot access vLLM model for pre-sync check")
        except Exception as e:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error checking vLLM weights BEFORE sync: {e}")
        
        torch.distributed.barrier()
        # ==============================================
        
        # Sync weights to vLLM model
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Syncing weights to vLLM with load_format={load_format}...")
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] Finished syncing weights to vLLM")
        
        log_gpu_memory_usage(f'[ShardingManager Rank {rank} TP {tp_rank}] After sync model weights', logger=logger)
        
        # Diagnose vLLM model after sync
        try:
            diagnose_embedding_weights(
                self.inference_engine.llm_engine.model,
                model_type="vLLM Model",
                prefix="AFTER SYNC",
                rank=rank,
                tp_rank=tp_rank
            )
        except Exception as e:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error diagnosing vLLM model after sync: {e}")
        
        # === PRINT 10: Check vLLM weights AFTER sync ===
        try:
            if hasattr(self.inference_engine.llm_engine, 'model'):
                vllm_model = self.inference_engine.llm_engine.model
                # Try common embedding attribute patterns
                vllm_emb_weight = None
                emb_attr_name = None
                
                for attr_name in ["embed_tokens.weight", "model.embed_tokens.weight", "transformer.wte.weight", "word_embeddings.weight"]:
                    try:
                        parts = attr_name.split(".")
                        curr_module = vllm_model
                        for part in parts[:-1]:
                            if hasattr(curr_module, part):
                                curr_module = getattr(curr_module, part)
                            else:
                                break
                        if hasattr(curr_module, parts[-1]):
                            vllm_emb_weight = getattr(curr_module, parts[-1])
                            emb_attr_name = attr_name
                            break
                    except:
                        continue
                
                if vllm_emb_weight is not None:
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight ('{emb_attr_name}') found AFTER sync")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Shape AFTER sync: {vllm_emb_weight.shape}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Mean AFTER sync: {vllm_emb_weight.mean().item():.6f}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Std AFTER sync: {vllm_emb_weight.std().item():.6f}")
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] vLLM Embedding Weight Device AFTER sync: {vllm_emb_weight.device}")
                    
                    # Check if weights are close to zero
                    if vllm_emb_weight.abs().mean().item() < 1e-6:
                        print(f"[ShardingManager Rank {rank} TP {tp_rank}] !!!!!!!! VLLM EMBEDDING WEIGHTS ARE ZERO AFTER SYNC !!!!!!!!")
                else:
                    print(f"[ShardingManager Rank {rank} TP {tp_rank}] Could not find vLLM embedding weights AFTER sync")
            else:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Cannot access vLLM model for post-sync check")
        except Exception as e:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error checking vLLM weights AFTER sync: {e}")
        
        torch.distributed.barrier()
        # =============================================
        
        # Unmerge LoRA adapter if using PEFT
        if is_peft_model:
            print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Unmerging adapter...")
            try:
                with FSDP.summon_full_params(self.module):
                    self.module._fsdp_wrapped_module.unmerge_adapter()
                    
                    # Diagnose after unmerge
                    diagnose_embedding_weights(
                        self.module._fsdp_wrapped_module, 
                        model_type="PEFT Model (after unmerge)", 
                        prefix="AFTER UNMERGE",
                        rank=rank, 
                        tp_rank=tp_rank
                    )
                    
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] LoRA Path: Adapter unmerged")
            except Exception as e:
                print(f"[ShardingManager Rank {rank} TP {tp_rank}] Error unmerging adapter: {e}")
            
        del params
        torch.cuda.empty_cache()
        
        log_gpu_memory_usage(f'[ShardingManager Rank {rank} TP {tp_rank}] After del state_dict and empty_cache', logger=logger)
        
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)
            
        print(f"[ShardingManager Rank {rank} TP {tp_rank}] === ENTER finished ===")

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        
        self.inference_engine.offload_model_weights()
        
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        data.batch = allgather_dict_tensors(data.batch.contiguous(),
                                            size=vllm_ps.get_tensor_model_parallel_world_size(),
                                            group=vllm_ps.get_tensor_model_parallel_group(),
                                            dim=0)

        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        broadcast_dict_tensor(data.batch,
                              src=vllm_ps.get_tensor_model_parallel_src_rank(),
                              group=vllm_ps.get_tensor_model_parallel_group())
        dp_rank = torch.distributed.get_rank()
        dp_size = torch.distributed.get_world_size()  # not consider torch micro-dp
        tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        if tp_size > 1:
            # TODO: shall we build a micro_dp group for vllm when integrating with vLLM?
            local_prompts = data.chunk(chunks=tp_size)
            data = local_prompts[dp_rank % tp_size]
        return data
