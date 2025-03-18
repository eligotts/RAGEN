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
        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        start_time = time.time()
        
        is_peft_model = isinstance(self.module._fsdp_wrapped_module, PeftModel)
        
        if is_peft_model:
            logger.warning(f"TIMING: Starting LoRA adapter merging process")
            merge_start = time.time()
            # the model to sync weights to is a vLLM model (not a peft model), so we need to merge the adapters
            with FSDP.summon_full_params(self.module):
                self.module.merge_adapter()
                merge_end = time.time()
                logger.warning(f"TIMING: LoRA adapter merging took {merge_end - merge_start:.4f} seconds")
                
                params_start = time.time()
                params = self.module._fsdp_wrapped_module.base_model.model.state_dict()
                params_end = time.time()
                logger.warning(f"TIMING: Getting state_dict after merge took {params_end - params_start:.4f} seconds")
            
            # FIXME: use more rigorous way to filter out the adapter weights
            filter_start = time.time()
            params = OrderedDict((k.replace(".base_layer.", "."), v) for k, v in params.items() if not ".lora_" in k)
            filter_end = time.time()
            logger.warning(f"TIMING: Filtering adapter params took {filter_end - filter_start:.4f} seconds")
        else:
            state_dict_start = time.time()
            params = self.module.state_dict()
            state_dict_end = time.time()
            logger.warning(f"TIMING: Regular state_dict() took {state_dict_end - state_dict_start:.4f} seconds")
        
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        
        # Copy, not share memory
        sync_start = time.time()
        load_format = 'hf' if self.full_params else 'dtensor'
        self.inference_engine.sync_model_weights(params, load_format=load_format)
        sync_end = time.time()
        logger.warning(f"TIMING: vLLM sync_model_weights took {sync_end - sync_start:.4f} seconds")
        
        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        if is_peft_model:
            unmerge_start = time.time()
            with FSDP.summon_full_params(self.module):
                self.module.unmerge_adapter()
            unmerge_end = time.time()
            logger.warning(f"TIMING: LoRA adapter unmerging took {unmerge_end - unmerge_start:.4f} seconds")
            
        cleanup_start = time.time()
        del params
        torch.cuda.empty_cache()
        cleanup_end = time.time()
        logger.warning(f"TIMING: Cleanup took {cleanup_end - cleanup_start:.4f} seconds")
        
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')
        total_time = time.time() - start_time
        logger.warning(f"TIMING: Total weight syncing process took {total_time:.4f} seconds")
        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        
        offload_start = time.time()
        self.inference_engine.offload_model_weights()
        offload_end = time.time()
        logger.warning(f"TIMING: vLLM weight offloading took {offload_end - offload_start:.4f} seconds")
        
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        cache_start = time.time()
        torch.cuda.empty_cache()
        cache_end = time.time()
        logger.warning(f"TIMING: CUDA cache clearing took {cache_end - cache_start:.4f} seconds")

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
