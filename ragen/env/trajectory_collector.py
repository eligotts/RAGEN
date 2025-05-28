"""
Trajectory Collector for coordinating LLM generation with environment execution.
This module handles the multi-turn rollout loop and trajectory data collection.
"""

import torch
import numpy as np
import uuid
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer

from .environment_manager import EnvironmentManagerBase


def torch_to_numpy(data, is_object=False):
    """Convert torch tensors to numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, bool)):
        return np.array(data)
    elif isinstance(data, (list, tuple)):
        if is_object:
            return np.array(data, dtype=object)
        else:
            return np.array(data)
    else:
        if is_object:
            return np.array(data, dtype=object)
        else:
            return np.array(data)


def to_list_of_dict(batch: DataProto) -> List[Dict]:
    """Convert DataProto batch to list of dictionaries."""
    batch_size = len(batch.batch['input_ids']) if 'input_ids' in batch.batch else len(batch.batch)
    
    result = []
    for i in range(batch_size):
        item = {}
        
        # Add tensor batch data
        for key, value in batch.batch.items():
            if isinstance(value, torch.Tensor):
                item[key] = value[i]
            else:
                item[key] = value[i] if hasattr(value, '__getitem__') else value
        
        # Add non-tensor batch data
        for key, value in batch.non_tensor_batch.items():
            if isinstance(value, np.ndarray) and value.dtype == object:
                item[key] = value[i]
            elif hasattr(value, '__getitem__') and len(value) > i:
                item[key] = value[i]
            else:
                item[key] = value
        
        result.append(item)
    
    return result


def filter_group_data(batch_list: List[Dict], episode_rewards: np.ndarray, 
                     episode_lengths: np.ndarray, success: Dict[str, np.ndarray],
                     traj_uid: np.ndarray, config, last_try: bool = False) -> Tuple:
    """
    Filter environment groups based on reward variance.
    Used for advanced algorithms like DAPO and GiGPO.
    """
    if not hasattr(config, 'algorithm') or not hasattr(config.algorithm, 'filter_groups'):
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    if not config.algorithm.filter_groups.enable:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    group_n = getattr(config.env.rollout, 'n', 1)
    if group_n <= 1:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    # Group filtering logic
    batch_size = len(batch_list) // group_n
    keep_indices = []
    
    for group_idx in range(batch_size):
        start_idx = group_idx * group_n
        end_idx = start_idx + group_n
        group_rewards = episode_rewards[start_idx:end_idx]
        
        # Keep groups with reward variance (not all identical)
        if last_try or not np.allclose(group_rewards, group_rewards[0]):
            keep_indices.extend(range(start_idx, end_idx))
    
    # Filter all data
    filtered_batch_list = [batch_list[i] for i in keep_indices]
    filtered_rewards = episode_rewards[keep_indices]
    filtered_lengths = episode_lengths[keep_indices]
    filtered_traj_uid = traj_uid[keep_indices]
    
    # Filter success metrics
    filtered_success = {}
    for key, value in success.items():
        filtered_success[key] = value[keep_indices]
    
    return filtered_batch_list, filtered_rewards, filtered_lengths, filtered_success, filtered_traj_uid


class TrajectoryCollector:
    """
    Coordinates LLM generation with environment execution for trajectory collection.
    """
    
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryCollector.
        
        Args:
            config: Configuration object containing data processing settings
            tokenizer: Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs (optional)
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor
    
    def preprocess_single_sample(self, item: int, gen_batch: DataProto, 
                                obs: Dict) -> Dict:
        """
        Process a single observation sample for model input.
        
        Args:
            item: Sample index in the batch
            gen_batch: Batch data containing original prompts
            obs: Environment observation
            
        Returns:
            Processed input data dict
        """
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        
        is_multi_modal = obs_image is not None
        
        # Convert anchor to numpy if needed
        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor
        
        # Build chat structure
        if obs_text is not None:
            # Use environment-provided text observation
            chat = np.array([{
                "content": obs_text,
                "role": "user",
            }])
        else:
            # Fallback to basic prompt
            chat = np.array([{
                "content": "What is your next action?",
                "role": "user",
            }])
        
        # Apply chat template
        prompt_with_chat_template = self.tokenizer.apply_chat_template(
            chat,
            add_generation_prompt=True,
            tokenize=False
        )
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data if present
        if is_multi_modal and self.processor is not None:
            # Handle multimodal processing (placeholder for now)
            row_dict['multi_modal_data'] = {'image': [obs_image]}
            # Add multimodal processing logic here if needed
        
        # Tokenize input
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=getattr(self.config.data, 'max_prompt_length', 2048),
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation='error'
        )
        
        # Compute position IDs
        position_ids = compute_position_id_with_mask(attention_mask)
        
        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'anchor_obs': _obs_anchor,
            'index': item,
        })
        
        # Add data source if available
        if hasattr(gen_batch, 'non_tensor_batch') and 'data_source' in gen_batch.non_tensor_batch:
            if item < len(gen_batch.non_tensor_batch['data_source']):
                row_dict['data_source'] = gen_batch.non_tensor_batch['data_source'][item]
            else:
                row_dict['data_source'] = 'default'
        else:
            row_dict['data_source'] = 'default'
        
        return row_dict
    
    def preprocess_batch(self, gen_batch: DataProto, obs: Dict) -> DataProto:
        """
        Process a batch of observation samples.
        
        Args:
            gen_batch: Batch data containing original prompts
            obs: Environment observation dictionary
            
        Returns:
            Processed batch data
        """
        # Determine batch size from observations
        obs_count = 0
        if obs.get('text') is not None:
            obs_count = len(obs['text'])
        elif obs.get('image') is not None:
            obs_count = len(obs['image'])
        elif obs.get('anchor') is not None:
            obs_count = len(obs['anchor'])
        else:
            raise ValueError("No valid observations found")
        
        processed_samples = []
        
        # Process each sample
        for item in range(obs_count):
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info if hasattr(gen_batch, 'meta_info') else {}
        )
        
        return new_batch
    
    def gather_rollout_data(self, total_batch_list: List[List[Dict]],
                           episode_rewards: np.ndarray, episode_lengths: np.ndarray,
                           success: Dict[str, np.ndarray], traj_uid: np.ndarray) -> DataProto:
        """
        Collect and organize trajectory data.
        
        Args:
            total_batch_list: List of trajectory data for each environment
            episode_rewards: Total rewards for each environment
            episode_lengths: Total steps for each environment
            success: Success metrics for each environment
            traj_uid: Trajectory unique identifiers
            
        Returns:
            Collected trajectory data
        """
        batch_size = len(total_batch_list)
        
        # Compute statistics
        episode_rewards_mean = np.mean(episode_rewards)
        episode_rewards_min = np.min(episode_rewards)
        episode_rewards_max = np.max(episode_rewards)
        
        episode_lengths_mean = np.mean(episode_lengths)
        episode_lengths_min = np.min(episode_lengths)
        episode_lengths_max = np.max(episode_lengths)
        
        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        # Collect effective batch data
        effective_batch = []
        for bs in range(batch_size):
            for data in total_batch_list[bs]:
                if data.get('active_masks', True):
                    # Add episode-level metrics
                    data['episode_rewards'] = episode_rewards[bs]
                    data['episode_rewards_mean'] = episode_rewards_mean
                    data['episode_rewards_min'] = episode_rewards_min
                    data['episode_rewards_max'] = episode_rewards_max
                    
                    data['episode_lengths'] = episode_lengths[bs]
                    data['episode_lengths_mean'] = episode_lengths_mean
                    data['episode_lengths_min'] = episode_lengths_min
                    data['episode_lengths_max'] = episode_lengths_max
                    
                    # Add success metrics
                    for key, value in success_rate.items():
                        data[key] = value
                    
                    effective_batch.append(data)
        
        # Convert to DataProto format
        if effective_batch:
            gen_batch_output = DataProto.from_single_dict(
                data=collate_fn(effective_batch)
            )
        else:
            # Create empty batch if no effective data
            gen_batch_output = DataProto.from_single_dict(
                data={'input_ids': torch.empty(0, 1, dtype=torch.long)}
            )
        
        return gen_batch_output
    
    def vanilla_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg,
                               envs: EnvironmentManagerBase) -> Tuple:
        """
        Standard multi-turn rollout loop.
        
        Args:
            gen_batch: Initial batch with prompts
            actor_rollout_wg: Actor model workers
            envs: Environment manager
            
        Returns:
            Trajectory data components
        """
        # Reset environments
        obs, infos = envs.reset()
        
        # Coordinate batch sizes
        obs_count = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        group_n = getattr(self.config.env.rollout, 'n', 1)
        
        if hasattr(gen_batch, 'batch') and len(gen_batch.batch) != obs_count and group_n > 0:
            gen_batch = gen_batch.repeat(repeat_times=group_n, interleave=True)
        
        batch_size = obs_count
        
        # Initialize tracking
        if group_n > 0:
            uid_batch = []
            for i in range(batch_size):
                if i % group_n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else:
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(batch_size)], dtype=object)
        
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.int32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        
        # Multi-turn interaction loop
        max_steps = getattr(self.config.env, 'max_steps', 50)
        for _step in range(max_steps):
            active_masks = np.logical_not(is_done)
            
            # Preprocess observations for model input
            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
            
            # Prepare input for model
            batch_input = batch.pop(
                batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                non_tensor_batch_keys=['anchor_obs', 'index', 'data_source'],
            )
            batch_input.meta_info = gen_batch.meta_info if hasattr(gen_batch, 'meta_info') else {}
            
            # Generate responses
            batch_output = actor_rollout_wg.generate_sequences(batch_input)
            
            # Add metadata
            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid
            
            # Combine input and output
            batch = batch.union(batch_output)
            
            # Decode responses to text actions
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            
            # Execute actions in environments
            next_obs, rewards, dones, infos = envs.step(text_actions)
            
            # Process rewards and dones
            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                dones = dones.squeeze(1)
            
            # Add action validity info
            if len(infos) > 0 and 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array(
                    [info['is_action_valid'] for info in infos], dtype=bool
                )
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)
            
            # Update episode tracking
            episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_lengths[active_masks] += 1
            
            # Add step data to batch
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Convert to list format and store
            batch_list = to_list_of_dict(batch)
            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i] if i < len(infos) else {})
            
            # Update done states
            is_done = np.logical_or(is_done, dones)
            
            # Update observations for next step
            obs = next_obs
            
            # Break if all environments are done
            if is_done.all():
                break
        
        # Evaluate success
        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )
        
        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid
    
    def dynamic_multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg,
                               envs: EnvironmentManagerBase) -> Tuple:
        """
        Dynamic rollout loop that continues until target batch size is met.
        Used for advanced algorithms like DAPO.
        
        Args:
            gen_batch: Initial batch
            actor_rollout_wg: Actor model workers
            envs: Environment manager
            
        Returns:
            Aggregated trajectory data
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        
        try_count = 0
        max_try_count = getattr(self.config.algorithm.filter_groups, 'max_num_gen_batches', 10)
        target_size = getattr(self.config.data, 'train_batch_size', 4) * getattr(self.config.env.rollout, 'n', 1)
        
        while len(total_batch_list) < target_size and try_count < max_try_count:
            if len(total_batch_list) > 0:
                print(f"Valid trajectories: {len(total_batch_list)}/{target_size}. "
                      f"Continuing generation... ({try_count}/{max_try_count})")
            
            try_count += 1
            
            # Run vanilla rollout
            batch_list, episode_rewards, episode_lengths, success, traj_uid = \
                self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
            
            # Filter data based on group criteria
            batch_list, episode_rewards, episode_lengths, success, traj_uid = \
                filter_group_data(
                    batch_list=batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    success=success,
                    traj_uid=traj_uid,
                    config=self.config,
                    last_try=(try_count == max_try_count),
                )
            
            # Accumulate results
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
        
        # Concatenate all results
        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {
            key: np.concatenate([success[key] for success in total_success], axis=0)
            for key in total_success[0].keys()
        }
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        
        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid
    
    def multi_turn_loop(self, gen_batch: DataProto, actor_rollout_wg,
                       envs: EnvironmentManagerBase, is_train: bool = True) -> DataProto:
        """
        Main entry point for multi-turn rollout.
        
        Args:
            gen_batch: Initial prompt batch
            actor_rollout_wg: Actor model workers
            envs: Environment manager
            is_train: Whether in training mode
            
        Returns:
            Final collected trajectory data
        """
        # Choose rollout strategy
        use_dynamic = (hasattr(self.config, 'algorithm') and 
                      hasattr(self.config.algorithm, 'filter_groups') and
                      self.config.algorithm.filter_groups.enable and is_train)
        
        if use_dynamic:
            # Dynamic sampling for advanced algorithms
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.dynamic_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        else:
            # Standard sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid = \
                self.vanilla_multi_turn_loop(gen_batch, actor_rollout_wg, envs)
        
        # Validate data consistency
        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        
        # Create final trajectory data
        gen_batch_output = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
        )
        
        return gen_batch_output 