"""
This is the environment state manager for the LLM agent.
author: Pingyue Zhang
date: 2025-03-30
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
import PIL.Image
import hydra
import random
import numpy as np

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

# Import parallel execution components
try:
    from ragen.env.parallel_env_container import MultiProcessEnvironmentContainer
    from ragen.env.projection_functions import create_projection_function
    PARALLEL_AVAILABLE = True
except ImportError:
    PARALLEL_AVAILABLE = False

@dataclass
class EnvStatus:
    """Status of an environment"""
    truncated: bool = False # done but not success
    terminated: bool = False # done and success
    num_actions: int = 0 # current action step (single action)
    rewards: List[float] = field(default_factory=list) # rewards for each turn
    seed: Optional[int] = None # what seed is used to reset this environment



class EnvStateManager:
    """Manager for the environment state
    The class is responsible for managing multiple (kinds of) environments
    
    """
    def __init__(self, config, mode: str = "train"):
        self.sys_config = config
        self.mode = mode
        self.config = getattr(self.sys_config.es_manager, mode)
        self.env_groups = int(self.config.env_groups)
        self.group_size = self.config.group_size
        
        # Determine if we should use parallel execution
        self.use_parallel = (
            PARALLEL_AVAILABLE and 
            getattr(self.sys_config, 'use_parallel_envs', True) and
            getattr(self.sys_config.es_manager, 'use_parallel_execution', True)
        )
        
        if self.use_parallel:
            print(f"EnvStateManager: Using parallel execution for {mode} environments")
            self._init_parallel_envs()
        else:
            print(f"EnvStateManager: Using single-process execution for {mode} environments")
            self._init_envs()
        
        self.rollout_cache = None

    def _init_envs(self):
        """Initialize the environments. train_envs and val_envs are lists of envs:
        Input: tags: ["SimpleSokoban", "HarderSokoban"]; n_groups: [1, 1]; group_size: 16
        Output: envs: List[Dict], each **entry** is a dict with keys: tag, group_id, env_id, env, env_config, status
        Example: [{"tag": "SimpleSokoban", "group_id": 0, "env_id": 0, "env": env, "config": env_config, "status": EnvStatus()},
            ...
            {"tag": "SimpleSokoban", "group_id": 0, "env_id": 15 (group_size - 1), ...},
            {"tag": "HarderSokoban", "group_id": 1, "env_id": 16, ...}
            ...]
        """
        assert sum(self.config.env_configs.n_groups) == self.env_groups, f"Sum of n_groups must equal env_groups. Got sum({self.config.env_configs.n_groups}) != {self.env_groups}"
        assert len(self.config.env_configs.tags) == len(self.config.env_configs.n_groups), f"Number of tags must equal number of n_groups. Got {len(self.config.env_configs.tags)} != {len(self.config.env_configs.n_groups)}"
        self.envs = self._init_env_instances(self.config)

    def _init_env_instances(self, config):
        env_list = []
        done_groups = 0
        for tag, n_group in zip(config.env_configs.tags, config.env_configs.n_groups):
            for env_id in range(done_groups * self.group_size, (done_groups + n_group) * self.group_size):
                cfg_template = self.sys_config.custom_envs[tag]
                env_class = cfg_template.env_type
                max_actions_per_traj = cfg_template.max_actions_per_traj
                if cfg_template.env_config is None:
                    env_config = REGISTERED_ENV_CONFIGS[env_class]()
                else:
                    env_config = REGISTERED_ENV_CONFIGS[env_class](**cfg_template.env_config)
                env_obj = REGISTERED_ENVS[env_class](env_config)
                entry = {'tag': tag, 'group_id': env_id // self.group_size, 'env_id': env_id, 
                        'env': env_obj, 'config': env_config, 'status': EnvStatus(), 'max_actions_per_traj': max_actions_per_traj}
                env_list.append(entry)
            done_groups += n_group
        return env_list

    def _init_parallel_envs(self):
        """Initialize parallel environments using MultiProcessEnvironmentContainer"""
        # For now, support single environment type (can be extended later)
        env_tag = self.config.env_configs.tags[0]
        env_type = self.sys_config.custom_envs[env_tag].env_type
        
        # Extract environment kwargs
        env_kwargs = self._extract_env_kwargs(env_tag)
        
        # Calculate seed based on mode
        base_seed = getattr(self.sys_config, 'seed', 42)
        if self.mode == "val":
            base_seed += 1000
        
        # Create parallel environment container
        self.parallel_container = MultiProcessEnvironmentContainer(
            env_type=env_type,
            env_num=self.env_groups,
            group_n=self.group_size,
            seed=base_seed,
            env_kwargs=env_kwargs
        )
        
        # Create projection function for action conversion
        self.projection_function = create_projection_function(env_type)
        
        # Store environment metadata for compatibility
        self.env_tag = env_tag
        self.env_type = env_type
        self.max_actions_per_traj = self.sys_config.custom_envs[env_tag].max_actions_per_traj
        
        # Create environment entries for compatibility with existing interface
        self.envs = []
        for env_id in range(self.env_groups * self.group_size):
            entry = {
                'tag': env_tag,
                'group_id': env_id // self.group_size,
                'env_id': env_id,
                'env': None,  # Not used in parallel mode
                'config': None,  # Not used in parallel mode
                'status': EnvStatus(),
                'max_actions_per_traj': self.max_actions_per_traj
            }
            self.envs.append(entry)
    
    def _extract_env_kwargs(self, env_tag: str) -> Dict[str, Any]:
        """Extract environment-specific configuration parameters."""
        env_config = self.sys_config.custom_envs[env_tag]
        env_kwargs = {}
        
        # Extract common parameters
        if hasattr(env_config, 'max_actions_per_traj'):
            env_kwargs['max_steps'] = env_config.max_actions_per_traj
        
        # Environment-specific parameters
        env_type = env_config.env_type
        
        if env_type == 'sokoban':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                sokoban_config = env_config.env_config
                if 'dim_room' in sokoban_config:
                    env_kwargs['dim_room'] = tuple(sokoban_config['dim_room'])
                if 'num_boxes' in sokoban_config:
                    env_kwargs['num_boxes'] = sokoban_config['num_boxes']
                if 'search_depth' in sokoban_config:
                    env_kwargs['search_depth'] = sokoban_config['search_depth']
                if 'max_steps' in sokoban_config:
                    env_kwargs['max_steps'] = sokoban_config['max_steps']
                if 'render_mode' in sokoban_config:
                    env_kwargs['render_mode'] = sokoban_config['render_mode']
        
        elif env_type == 'webshop':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                webshop_config = env_config.env_config
                for key, value in webshop_config.items():
                    env_kwargs[key] = value
        
        elif env_type == 'countdown':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                countdown_config = env_config.env_config
                for key, value in countdown_config.items():
                    env_kwargs[key] = value
        
        elif env_type == 'metamathqa':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                metamath_config = env_config.env_config
                for key, value in metamath_config.items():
                    env_kwargs[key] = value
        
        elif env_type == 'frozen_lake':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                frozen_lake_config = env_config.env_config
                for key, value in frozen_lake_config.items():
                    env_kwargs[key] = value
        
        elif env_type == 'bandit':
            if hasattr(env_config, 'env_config') and env_config.env_config:
                bandit_config = env_config.env_config
                for key, value in bandit_config.items():
                    env_kwargs[key] = value
        
        return env_kwargs

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
        if self.use_parallel:
            return self._reset_parallel(seed)
        else:
            return self._reset_single_process(seed)
    
    def _reset_parallel(self, seed: Optional[int] = None):
        """Reset environments using parallel execution"""
        # Calculate seed
        if self.mode == "train":
            reset_seed = random.randint(0, 1000000) if seed is None else seed
        else:
            reset_seed = 123
        
        # Reset all parallel environments
        raw_obs, infos = self.parallel_container.reset()
        
        # Initialize rollout cache
        rollout_cache = []
        for env_id in range(len(self.envs)):
            cache = {
                "env_id": env_id,
                "history": [],
                "group_id": env_id // self.group_size,
                "tag": self.env_tag,
                "penalty": 0
            }
            rollout_cache.append(cache)
        
        # Reset environment status
        for entry in self.envs:
            entry['status'] = EnvStatus(seed=reset_seed)
        
        # Update rollout cache with initial observations
        for cache, obs in zip(rollout_cache, raw_obs):
            next_state = self._handle_mm_state(obs)
            cache['history'] = self._update_cache_history(
                cache['history'], 
                next_state=next_state, 
                actions_left=self.max_actions_per_traj, 
                num_actions_info=None
            )
        
        self.rollout_cache = rollout_cache
        return rollout_cache
    
    def _reset_single_process(self, seed: Optional[int] = None):
        """Reset environments using single-process execution (original implementation)"""
        def _expand_seed(seed: int):
            seeds = [[seed + i] * self.group_size for i in range(self.env_groups)] # [[seed, ..., seed], [seed+1, ..., seed+1], ...]
            return sum(seeds, [])

        envs = self.envs
        rollout_cache = [{"env_id": entry['env_id'], "history": [], "group_id": entry['group_id'], "tag": entry['tag'], "penalty": 0} for entry in envs]

        # reset all environments
        if self.mode == "train":
            seed = random.randint(0, 1000000) if seed is None else seed # get a random seed
        else:
            seed = 123
        seeds = _expand_seed(seed)
        for seed, entry in zip(seeds, envs):
            entry['env'].reset(seed=seed, mode=self.mode)
            entry['status'] = EnvStatus(seed=seed)

        # update rollout cache
        for cache, env in zip(rollout_cache, envs):
            next_state = self._handle_mm_state(env['env'].render())
            cache['history'] = self._update_cache_history(cache['history'], next_state=next_state, actions_left=env['max_actions_per_traj'], num_actions_info=None)
            
        self.rollout_cache = rollout_cache
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments.
        1. extract valid actions from the action lookup table (if exists) and execute the actions, and update rollout cache
        2. Since rollout does not need to act over done envs, whenever the environment is done, we only update rollout cache, but not output env_outputs.
        Input:
        all_env_inputs: List[Dict]
            {env_id: int, llm_response: str, actions: List[str]}
            NOTE: should use env_id as index for existing some already done envs
        env_outputs: List[Dict]
            {env_id: int, history: List[Dict][{state: str, actions: List[str], reward: float, info: Dict, llm_response: str, llm_raw_response: str, (Optional)images: List[PIL.Image.Image]}]}
        """
        if self.use_parallel:
            return self._step_parallel(all_env_inputs)
        else:
            return self._step_single_process(all_env_inputs)
    
    def _step_parallel(self, all_env_inputs: List[Dict]):
        """Step environments using parallel execution"""
        # Prepare actions for all environments (fill with None for inactive ones)
        num_envs = len(self.envs)
        text_actions = [None] * num_envs
        env_input_map = {}
        
        # Map input actions to environment indices
        for env_input in all_env_inputs:
            env_id = env_input['env_id']
            # Use the actions list from env_input, not llm_response
            actions_list = env_input.get('actions', [])
            if actions_list:
                # For now, take the first action (can be extended for multi-action)
                text_actions[env_id] = actions_list[0]
            else:
                text_actions[env_id] = ""
            env_input_map[env_id] = env_input
        
        # Create full action list for parallel execution (only for active environments)
        full_text_actions = []
        active_indices = []
        for i, action in enumerate(text_actions):
            if action is not None:
                full_text_actions.append(action)
                active_indices.append(i)
        
        if not full_text_actions:
            return []  # No active environments
        
        # Project text actions to environment actions
        projected_actions, validity_flags = self.projection_function(full_text_actions)
        
        # For bandit environments, convert arm names to action IDs using ACTION_LOOKUP
        if self.env_type == 'bandit':
            projected_actions = self._convert_bandit_actions_to_ids(projected_actions, active_indices)
        
        # Create full action arrays for parallel container (pad with defaults for inactive envs)
        full_projected_actions = [""] * num_envs
        full_validity_flags = [False] * num_envs
        
        for i, env_id in enumerate(active_indices):
            full_projected_actions[env_id] = projected_actions[i]
            full_validity_flags[env_id] = validity_flags[i]
        
        # Execute actions in parallel environments
        raw_obs, rewards, dones, infos = self.parallel_container.step(full_projected_actions)
        
        # Process results and update rollout cache
        env_outputs = []
        
        for i, env_id in enumerate(active_indices):
            if env_id >= len(self.envs):
                continue
                
            entry = self.envs[env_id]
            env_input = env_input_map[env_id]
            
            # Get results for this environment
            obs = raw_obs[env_id]
            reward = rewards[env_id]
            done = dones[env_id]
            info = infos[env_id] if env_id < len(infos) else {}
            is_valid = full_validity_flags[env_id]
            
            # Update penalty for invalid actions
            original_actions = env_input.get('actions', [])
            if not is_valid or len(original_actions) == 0:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
            
            # Update environment status
            status = entry['status']
            actions_left_before = entry['max_actions_per_traj'] - status.num_actions
            
            # Process executed actions (limit to actions_left_before)
            executed_actions = [full_projected_actions[env_id]] if full_projected_actions[env_id] and actions_left_before > 0 else []
            
            # Update status
            status.num_actions += len(executed_actions)
            status.rewards.append(reward)
            
            # Check if environment is done
            turn_done = done
            if turn_done:
                status.terminated = True
                status.truncated = not info.get('success', False)
            
            # Check if max actions reached
            if status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
                status.truncated = True
                status.terminated = True
                turn_done = True
            
            # Update rollout cache history
            next_state = self._handle_mm_state(obs)
            actions_left = entry['max_actions_per_traj'] - status.num_actions
            
            history = self._update_cache_history(
                self.rollout_cache[env_id]['history'],
                next_state=next_state,
                actions_left=actions_left,
                num_actions_info={
                    'actions': executed_actions,
                    'reward': reward,
                    'info': info,
                    'llm_response': env_input['llm_response'],
                    'llm_raw_response': env_input['llm_raw_response']
                }
            )
            
            self.rollout_cache[env_id]['history'] = history
            
            # Add to outputs if not done (for continued LLM generation)
            if not turn_done:
                env_outputs.append(self.rollout_cache[env_id])
        
        return env_outputs
    
    def _step_single_process(self, all_env_inputs: List[Dict]):
        """Step environments using single-process execution (original implementation)"""
        def _execute_actions(env, actions):
            acc_reward, turn_info, turn_done = 0, {}, False
            executed_actions = []
            for action in actions:
                _, reward, done, info = env.step(action)
                acc_reward += reward
                turn_info.update(info) # NOTE: currently use last info for multi-action
                executed_actions.append(action)
                if done:
                    turn_done = True
                    break
            
            return acc_reward, turn_info, turn_done, executed_actions

        def _log_env_state(status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
            obs = self._handle_mm_state(cur_obs)
            status.num_actions += len(executed_actions)
            status.rewards.append(acc_reward) # NOTE use turn-wise acc_reward
            actions_left = max_actions_per_traj - status.num_actions
            if turn_done:
                status.terminated = True # TODO check terminated definition in gymnasium
                status.truncated = not turn_info.get('success', False)
            history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
                'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
                'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
            })
            # filter out invalid actions
            # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
            return status, history

        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = _execute_actions(env, valid_actions[:actions_left_before])
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = _log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
            entry['status'] = status
            if entry['status'].num_actions >= entry['max_actions_per_traj'] and not turn_done:
                entry['status'].truncated = True
                entry['status'].terminated = True
                turn_done = True
            self.rollout_cache[env_id]['history'] = history
            if not turn_done: # NOTE done environments are not sent for further llm generation (for efficiency)
                env_outputs.append(self.rollout_cache[env_id])

        return env_outputs

    def get_rollout_states(self):
        """Get the final output for all environment"""
        envs = self.envs
        rollout_cache = self.rollout_cache
        TURN_LVL_METRICS = ['action_is_effective', 'action_is_valid', 'end_of_page']

        # add metrics to rollout cache
        for entry, cache in zip(envs, rollout_cache):
            status = entry['status']
            env_metric = {
                'success': float(status.terminated and (not status.truncated)),
                'num_actions': status.num_actions,
            }
            custom_metric = {}
            for turn in cache['history']:
                for k, v in turn.get('info', {}).items():
                    if k == 'success':
                        continue
                    if k not in custom_metric:
                        custom_metric[k] = []
                    custom_metric[k].append(float(v))
            for k, v in custom_metric.items():
                # TODO: Move TURN_LVL_METRICS into the environment
                if "Webshop" not in k or ("Webshop" in k and k in TURN_LVL_METRICS):
                    env_metric[k] = np.sum(v) / (len(cache['history']) - 1) # NOTE: exclude the last observation
                else:
                    env_metric[k] = np.sum(v)


            cache['history'][-1]['metrics'] = custom_metric
            env_metric = {f"{entry['tag']}/{k}": v for k, v in env_metric.items()}
            cache['metrics'] = env_metric
            if entry['tag'] == "MetamathQA":
                cache['correct_answer'] = entry['env'].correct_answer
        return rollout_cache




    def _update_cache_history(self, history: List[Dict], next_state, actions_left, num_actions_info: Optional[Dict] = None):
        """
        Update last step info and append state to history
        """
        if num_actions_info is not None: # update last step info
            assert len(history), "History should not be empty"
            history[-1].update(num_actions_info)
        
        entry = {} # append state to history
        if isinstance(next_state, str): # text state
            entry['state'] = next_state
        else: # multimodal state
            entry['state'] = "<images>" * len(next_state)
            entry['images'] = next_state
        entry['actions_left'] = actions_left
        history.append(entry)
        return history

    def _extract_map_valid_actions(self, entry: Dict, actions: List[str]):
        """extract valid actions from the action lookup table (if exists)"""
        mapped_actions = []
        action_lookup = getattr(entry['env'].config, 'action_lookup', None)
        if action_lookup is None:
            mapped_actions = actions
        else: # the envs have pre-defined action lookup
            rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
            actions = [action.lower() for action in actions]
            mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
        return mapped_actions
    
    def _handle_mm_state(self, state: Union[str, np.ndarray, list[np.ndarray]]):
        """Handle the state from the environment
        """
        if isinstance(state, str): # text state
            return state
        elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
            state = [state]
        results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
        return results
        
    def render(self):
        if self.use_parallel:
            return self.parallel_container.render()
        else:
            rendered_list = [entry['env'].render() for entry in self.envs]
            return rendered_list

    def close(self):
        if self.use_parallel:
            if hasattr(self, 'parallel_container'):
                self.parallel_container.close()
        else:
            for entry in self.envs:
                entry['env'].close()

    def _convert_bandit_actions_to_ids(self, arm_names: List[str], active_indices: List[int]) -> List[int]:
        """Convert bandit arm names to action IDs using environment ACTION_LOOKUP"""
        action_ids = []
        
        # Get action lookup from the first active environment (they should all have the same mapping)
        if active_indices and hasattr(self, 'parallel_container'):
            try:
                # Query the first active environment for its ACTION_LOOKUP
                env_idx = active_indices[0]
                if env_idx < len(self.parallel_container.parent_remotes):
                    remote = self.parallel_container.parent_remotes[env_idx]
                    remote.send(('get_action_lookup', None))
                    status, action_lookup = remote.recv()
                    
                    if status == 'success' and action_lookup:
                        # Create reverse lookup: arm_name -> action_id
                        name_to_id = {v.lower(): k for k, v in action_lookup.items()}
                        
                        # Convert each arm name to action ID
                        for arm_name in arm_names:
                            arm_name_lower = arm_name.lower()
                            if arm_name_lower in name_to_id:
                                action_ids.append(name_to_id[arm_name_lower])
                            else:
                                # Default to first action if not found
                                action_ids.append(min(action_lookup.keys()))
                    else:
                        # Fallback: use default action IDs
                        action_ids = [1] * len(arm_names)
                else:
                    # Fallback: use default action IDs
                    action_ids = [1] * len(arm_names)
            except Exception as e:
                print(f"Error getting action lookup: {e}")
                # Fallback: use default action IDs
                action_ids = [1] * len(arm_names)
        else:
            # Fallback: use default action IDs
            action_ids = [1] * len(arm_names)
        
        return action_ids




@hydra.main(version_base=None, config_path="../../config", config_name="base")
def main(config):
    """
    Unit test for EnvStateManager
    """
    es_manager = EnvStateManager(config, mode="train")
    print("Initializing environments...")
    es_manager.reset(seed=123)

    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRunning step for training environments...")
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")

    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go left, go up",
            "llm_response": "Go left, go up",
            "actions": ["left", "up"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Go up, go up",
            "llm_response": "Go up, go up",
            "actions": ["up", "up", "up", "up", "up"]
        }
    ]
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Active environments after step: {len(env_outputs)}")
    print(f"env_outputs[:2]: {env_outputs[:2]}")
    
    renders = es_manager.render()
    for i, render in enumerate(renders[:4]):  # Show first 2 environments
        print(f"Environment {i}:\n{render}\n")
    
    print("\nRendering final output...")
    final_outputs = es_manager.get_rollout_states()
    print(f"final outputs[:4]: {final_outputs[:4]}")
    
    print("\nClosing environments...")
    es_manager.close()
    print("Test completed successfully!")


if __name__ == "__main__":
	main()
