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
import os
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

# Global variables for worker processes
_worker_envs = {}
_worker_format_penalty = 0

def _worker_init(env_recreation_data, format_penalty):
    """Initialize environments in each worker process (called once per worker)"""
    global _worker_envs, _worker_format_penalty
    _worker_format_penalty = format_penalty
    _worker_envs = {}
    
    # Pre-create all environments in this worker
    for env_id, env_data in env_recreation_data.items():
        try:
            env_class_name = env_data['env_class']
            env_config_class = REGISTERED_ENV_CONFIGS[env_class_name]
            env_class = REGISTERED_ENVS[env_class_name]
            
            # Recreate config object
            env_config = env_config_class(**env_data['env_config_dict'])
            
            # Create environment
            env_obj = env_class(env_config)
            
            # Reset with the same seed as original
            # Note: some environments (like WebShop) may need special handling
            if env_class_name == 'webshop':
                # WebShop may need special initialization in worker processes
                try:
                    env_obj.reset(seed=env_data['seed'], mode=env_data.get('mode'))
                except Exception as webshop_error:
                    print(f"Warning: WebShop env {env_id} failed to reset in worker: {webshop_error}")
                    # Still store the environment, but mark it as potentially problematic
                    _worker_envs[env_id] = {
                        'env': env_obj,
                        'config': env_config,
                        'max_actions_per_traj': env_data['max_actions_per_traj'],
                        'initialization_warning': True
                    }
                    continue
            else:
                env_obj.reset(seed=env_data['seed'], mode=env_data.get('mode'))
            
            _worker_envs[env_id] = {
                'env': env_obj,
                'config': env_config,
                'max_actions_per_traj': env_data['max_actions_per_traj']
            }
        except Exception as e:
            print(f"Failed to initialize env {env_id} ({env_data.get('env_class', 'unknown')}) in worker: {e}")
            _worker_envs[env_id] = None

def _extract_map_valid_actions_worker(env_config, actions):
    """Worker version of _extract_map_valid_actions"""
    mapped_actions = []
    action_lookup = getattr(env_config, 'action_lookup', None)
    if action_lookup is None:
        mapped_actions = actions
    else:
        rev_action_lookup = {v.lower(): k for k, v in action_lookup.items()}
        actions = [action.lower() for action in actions]
        mapped_actions = [rev_action_lookup[action] for action in actions if action in rev_action_lookup]
    return mapped_actions

def _execute_actions_worker(env, actions):
    """Worker version of _execute_actions"""
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

def _handle_mm_state_worker(state):
    """Worker version of _handle_mm_state"""
    if isinstance(state, str): # text state
        return state
    elif isinstance(state, np.ndarray): # when env state is a single image, convert it to a list to unify output format
        state = [state]
    results = [PIL.Image.fromarray(_state, mode='RGB') for _state in state]
    return results

def _worker_step_single_env(env_input, current_status_dict):
    """
    Process a single environment step in worker process
    Reuses existing logic from the main process
    """
    global _worker_envs, _worker_format_penalty
    
    env_id = env_input['env_id']
    
    if env_id not in _worker_envs or _worker_envs[env_id] is None:
        return {
            'env_id': env_id,
            'error': f'Environment {env_id} not available in worker',
            'success': False
        }
    
    try:
        worker_env_data = _worker_envs[env_id]
        env = worker_env_data['env']
        env_config = worker_env_data['config'] 
        max_actions_per_traj = worker_env_data['max_actions_per_traj']
        
        # Recreate current status from dict
        current_status = EnvStatus()
        current_status.num_actions = current_status_dict['num_actions']
        current_status.rewards = current_status_dict['rewards'].copy()
        current_status.truncated = current_status_dict['truncated']
        current_status.terminated = current_status_dict['terminated']
        current_status.seed = current_status_dict['seed']
        
        # === REUSE EXISTING LOGIC ===
        actions_left_before = max_actions_per_traj - current_status.num_actions
        
        # Extract valid actions (reuse existing logic)
        valid_actions = _extract_map_valid_actions_worker(env_config, env_input['actions'])
        
        # Execute actions (reuse existing logic)  
        acc_reward, turn_info, turn_done, executed_actions = _execute_actions_worker(env, valid_actions[:actions_left_before])
        
        # Calculate penalty
        penalty = 0
        if len(valid_actions) != len(env_input['actions']) or not valid_actions:
            penalty = _worker_format_penalty
            
        # Update status
        current_status.num_actions += len(executed_actions)
        current_status.rewards.append(acc_reward)
        actions_left = max_actions_per_traj - current_status.num_actions
        if turn_done:
            current_status.terminated = True
            current_status.truncated = not turn_info.get('success', False)
        if current_status.num_actions >= max_actions_per_traj and not turn_done:
            current_status.truncated = True
            current_status.terminated = True
            turn_done = True
            
        # Get next state
        next_state_raw = env.render()
        next_state = _handle_mm_state_worker(next_state_raw)
        
        return {
            'env_id': env_id,
            'success': True,
            'status_updates': {
                'num_actions': current_status.num_actions,
                'rewards': current_status.rewards,
                'truncated': current_status.truncated,
                'terminated': current_status.terminated,
                'seed': current_status.seed
            },
            'next_state': next_state,
            'executed_actions': executed_actions,
            'valid_actions': valid_actions,
            'acc_reward': acc_reward,
            'turn_done': turn_done,
            'turn_info': turn_info,
            'penalty': penalty,
            'actions_left': actions_left
        }
        
    except Exception as e:
        return {
            'env_id': env_id,
            'error': str(e),
            'success': False
        }

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
        self._init_envs()
        self.rollout_cache = None
        
        # Initialize multiprocessing components
        self.use_multiprocessing = getattr(self.sys_config.es_manager, 'use_multiprocessing', True)
        self.max_workers = getattr(self.sys_config.es_manager, 'max_workers', min(8, os.cpu_count()))
        self.process_pool = None
        self.env_recreation_data = None
        if self.use_multiprocessing:
            self._init_process_pool()

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

    def reset(self, seed: Optional[int] = None):
        """
        Reset the environments and get initial observation
        build up rollout cache like [{"env_id": int, "history": List[Dict], "group_id": int}, ...]
        """
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
        
        # Reinitialize process pool with new environment states
        if self.use_multiprocessing and self.process_pool is not None:
            try:
                # Shutdown old pool
                self.process_pool.shutdown(wait=False)
                # Reinitialize with new states
                self._init_process_pool()
            except Exception as e:
                print(f"Failed to reinitialize process pool after reset: {e}")
                self.use_multiprocessing = False
                self.process_pool = None
        
        return rollout_cache

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments with optional multiprocessing"""
        if not self.use_multiprocessing or self.process_pool is None:
            return self._step_sequential(all_env_inputs)
        
        try:
            return self._step_parallel(all_env_inputs)
        except Exception as e:
            print(f"Parallel processing failed: {e}, falling back to sequential")
            return self._step_sequential(all_env_inputs)

    def _step_sequential(self, all_env_inputs: List[Dict]):
        """Original sequential implementation (UNCHANGED)"""
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

    def _step_parallel(self, all_env_inputs: List[Dict]):
        """New parallel implementation"""
        if not all_env_inputs:
            return []
        
        # Check if any environment types should force sequential processing
        sequential_env_types = getattr(self.sys_config.es_manager, 'sequential_env_types', [])
        if sequential_env_types:
            for env_input in all_env_inputs:
                env_id = env_input['env_id']
                entry = self.envs[env_id]
                env_class_name = None
                for name, cls in REGISTERED_ENVS.items():
                    if isinstance(entry['env'], cls):
                        env_class_name = name
                        break
                if env_class_name in sequential_env_types:
                    print(f"Environment type {env_class_name} is configured for sequential processing, falling back...")
                    return self._step_sequential(all_env_inputs)
        
        # Prepare current status data for workers
        status_data = {}
        for env_input in all_env_inputs:
            env_id = env_input['env_id']
            entry = self.envs[env_id]
            status_data[env_id] = {
                'num_actions': entry['status'].num_actions,
                'rewards': entry['status'].rewards.copy(),
                'truncated': entry['status'].truncated,
                'terminated': entry['status'].terminated,
                'seed': entry['status'].seed
            }
        
        # Submit all work to process pool
        futures = []
        for env_input in all_env_inputs:
            future = self.process_pool.submit(
                _worker_step_single_env,
                env_input,
                status_data[env_input['env_id']]
            )
            futures.append((env_input['env_id'], future))
        
        # Collect results and update main process state
        env_outputs = []
        for env_id, future in futures:
            try:
                result = future.result(timeout=30)  # 30 second timeout per env
                if result['success']:
                    self._apply_worker_result(env_id, result, all_env_inputs)
                    
                    # Add to outputs if environment is not done
                    if not result['turn_done']:
                        env_outputs.append(self.rollout_cache[env_id])
                else:
                    print(f"Worker failed for env {env_id}: {result.get('error', 'Unknown error')}")
                    # Could fall back to sequential for this specific env
                    
            except Exception as e:
                print(f"Failed to get result for env {env_id}: {e}")
        
        return env_outputs

    def _apply_worker_result(self, env_id, result, all_env_inputs):
        """Apply worker results to main process state"""
        # Find the corresponding env_input
        env_input = next(inp for inp in all_env_inputs if inp['env_id'] == env_id)
        
        # Update environment status
        entry = self.envs[env_id]
        status = entry['status']
        status.num_actions = result['status_updates']['num_actions']
        status.rewards = result['status_updates']['rewards']
        status.truncated = result['status_updates']['truncated']
        status.terminated = result['status_updates']['terminated']
        
        # Update rollout cache penalty
        if result['penalty'] > 0:
            self.rollout_cache[env_id]["penalty"] += result['penalty']
        
        # Update history using existing logic
        self.rollout_cache[env_id]['history'] = self._update_cache_history(
            self.rollout_cache[env_id]['history'],
            next_state=result['next_state'],
            actions_left=result['actions_left'],
            num_actions_info={
                'actions': result['executed_actions'],
                'reward': result['acc_reward'],
                'info': result['turn_info'],
                'llm_response': env_input['llm_response'],
                'llm_raw_response': env_input['llm_raw_response']
            }
        )

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
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def close(self):
        """Enhanced close method with process pool cleanup"""
        # Close environments
        for entry in self.envs:
            entry['env'].close()
        
        # Close process pool
        if hasattr(self, 'process_pool') and self.process_pool is not None:
            try:
                self.process_pool.shutdown(wait=True, timeout=10)
            except Exception as e:
                print(f"Error shutting down process pool: {e}")
            finally:
                self.process_pool = None

    def _create_env_recreation_data(self):
        """Create minimal data needed to recreate environments in workers"""
        recreation_data = {}
        for entry in self.envs:
            # Get the environment class name from the registered environments
            env_class_name = None
            for name, cls in REGISTERED_ENVS.items():
                if isinstance(entry['env'], cls):
                    env_class_name = name
                    break
            
            if env_class_name is None:
                raise ValueError(f"Could not find environment class for env_id {entry['env_id']}")
            
            recreation_data[entry['env_id']] = {
                'tag': entry['tag'],
                'env_class': env_class_name,
                'env_config_dict': entry['config'].__dict__.copy(),
                'max_actions_per_traj': entry['max_actions_per_traj'],
                'seed': entry['status'].seed,
                'mode': self.mode
            }
        return recreation_data

    def _init_process_pool(self):
        """Initialize the persistent process pool"""
        try:
            # Create recreation data once
            self.env_recreation_data = self._create_env_recreation_data()
            
            # Initialize the process pool with environment setup
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.max_workers,
                initializer=_worker_init,
                initargs=(self.env_recreation_data, self.sys_config.es_manager.format_penalty)
            )
        except Exception as e:
            print(f"Failed to initialize process pool: {e}, falling back to sequential processing")
            self.use_multiprocessing = False
            self.process_pool = None




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
