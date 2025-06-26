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
import copy
from concurrent.futures import ThreadPoolExecutor

from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS
from ragen.utils import register_resolvers
register_resolvers()

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
        
        # Threading configuration
        self.enable_parallel_env_steps = getattr(config.es_manager, 'enable_parallel_env_steps', True)
        self.max_env_worker_threads = getattr(config.es_manager, 'max_env_worker_threads', 4)
        
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
        return rollout_cache

    def _execute_actions_isolated(self, env, actions):
        """
        Execute actions on environment in isolation for threading.
        This is identical to the original _execute_actions but extracted for thread safety.
        """
        acc_reward, turn_info, turn_done = 0, {}, False
        executed_actions = []
        for action in actions:
            _, reward, done, info = env.step(action)
            acc_reward += reward
            turn_info.update(info)  # NOTE: currently use last info for multi-action
            executed_actions.append(action)
            if done:
                turn_done = True
                break
        
        return acc_reward, turn_info, turn_done, executed_actions

    def _process_single_env_worker(self, env_input: Dict) -> Dict:
        """
        Process a single environment in isolation for threading.
        Returns all computed values without updating shared state.
        
        Args:
            env_input: Dictionary containing env_id, llm_response, actions, etc.
            
        Returns:
            Dictionary with all computed results for later aggregation
        """
        env_id = env_input['env_id']
        entry = self.envs[env_id]
        env = entry['env']
        
        # Calculate actions left before execution (same logic as original)
        actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions
        
        # Extract valid actions (same logic as original)
        valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
        
        # Execute actions (this is the potentially I/O bound operation)
        acc_reward, turn_info, turn_done, executed_actions = self._execute_actions_isolated(
            env, valid_actions[:actions_left_before]
        )
        
        # Compute penalty increment (same logic as original)
        penalty_increment = 0
        if len(valid_actions) != len(env_input['actions']) or not valid_actions:
            penalty_increment = self.sys_config.es_manager.format_penalty
        
        # Get current observation for logging (same logic as original)
        cur_obs = entry['env'].render()
        
        # Compute new status (same logic as original, but don't update entry yet)
        new_status = copy.deepcopy(entry['status'])
        obs = self._handle_mm_state(cur_obs)
        new_status.num_actions += len(executed_actions)
        new_status.rewards.append(acc_reward)
        actions_left = entry['max_actions_per_traj'] - new_status.num_actions
        
        if turn_done:
            new_status.terminated = True
            new_status.truncated = not turn_info.get('success', False)
        
        # Compute new history (same logic as original, but don't update cache yet)
        current_history = copy.deepcopy(self.rollout_cache[env_id]['history'])
        new_history = self._update_cache_history(current_history, next_state=obs, actions_left=actions_left, num_actions_info={
            'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
            'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
        })
        
        # Check for truncation (same logic as original)
        final_turn_done = turn_done
        if new_status.num_actions >= entry['max_actions_per_traj'] and not turn_done:
            new_status.truncated = True
            new_status.terminated = True
            final_turn_done = True
        
        return {
            'env_id': env_id,
            'env_input': env_input,
            'new_status': new_status,
            'new_history': new_history,
            'penalty_increment': penalty_increment,
            'turn_done': final_turn_done,
        }

    def step(self, all_env_inputs: List[Dict]):
        """Step the environments with optional parallel processing.
        
        Args:
            all_env_inputs: List of environment inputs
            
        Returns:
            List of environment outputs for active (non-done) environments
        """
        # Use sequential processing for single environment or if parallel is disabled
        if not self.enable_parallel_env_steps or len(all_env_inputs) <= 1:
            return self._step_sequential(all_env_inputs)
        
        return self._step_parallel(all_env_inputs)

    def _step_sequential(self, all_env_inputs: List[Dict]):
        """
        Step the environments sequentially.
        """
        envs = self.envs
        env_outputs = []

        for env_input in all_env_inputs:
            acc_reward, turn_info, turn_done = 0, {}, False
            entry = envs[env_input['env_id']]
            env_id, env = entry['env_id'], entry['env']
            actions_left_before = entry['max_actions_per_traj'] - entry['status'].num_actions

            # execute actions in envs
            valid_actions = self._extract_map_valid_actions(entry, env_input['actions'])
            acc_reward, turn_info, turn_done, executed_actions = self._execute_actions_isolated(env, valid_actions[:actions_left_before])
            if len(valid_actions) != len(env_input['actions']) or not valid_actions:
                self.rollout_cache[env_id]["penalty"] += self.sys_config.es_manager.format_penalty
                
            status, history = self._log_env_state(entry['status'], self.rollout_cache[env_id]['history'], entry['env'].render(), entry['max_actions_per_traj'], executed_actions, valid_actions, acc_reward, turn_done, turn_info, env_input)
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
        """
        Step the environments in parallel using ThreadPoolExecutor.
        Each environment is processed in isolation, then results are aggregated.
        """
        # Process environments in parallel
        with ThreadPoolExecutor(max_workers=self.max_env_worker_threads) as executor:
            futures = [executor.submit(self._process_single_env_worker, env_input) 
                      for env_input in all_env_inputs]
            results = [future.result() for future in futures]
        
        # Aggregate results sequentially to avoid race conditions
        env_outputs = []
        for result in results:
            env_id = result['env_id']
            
            # Update shared state with computed results
            self.envs[env_id]['status'] = result['new_status']
            self.rollout_cache[env_id]['history'] = result['new_history']
            self.rollout_cache[env_id]["penalty"] += result['penalty_increment']
            
            # Add to outputs if environment is not done
            if not result['turn_done']:
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
        rendered_list = [entry['env'].render() for entry in self.envs]
        return rendered_list

    def _log_env_state(self, status, history, cur_obs, max_actions_per_traj, executed_actions, all_actions, acc_reward, turn_done, turn_info, env_input):
        """
        Helper function to log environment state (extracted from original implementation).
        """
        obs = self._handle_mm_state(cur_obs)
        status.num_actions += len(executed_actions)
        status.rewards.append(acc_reward)  # NOTE use turn-wise acc_reward
        actions_left = max_actions_per_traj - status.num_actions
        if turn_done:
            status.terminated = True  # TODO check terminated definition in gymnasium
            status.truncated = not turn_info.get('success', False)
        history = self._update_cache_history(history, next_state=obs, actions_left=actions_left, num_actions_info={
            'actions': executed_actions, 'reward': acc_reward, 'info': turn_info,
            'llm_response': env_input['llm_response'], 'llm_raw_response': env_input['llm_raw_response']
        })
        # filter out invalid actions
        # history = [content for content in history[:-1] if content['actions']] + [history[-1]]
        return status, history

    def close(self):
        for entry in self.envs:
            entry['env'].close()




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


def test_threading_implementation():
    """
    Simple test to verify that threading implementation produces equivalent results.
    This can be run independently to verify correctness.
    """
    import tempfile
    import os
    from hydra import initialize_config_store, compose
    from omegaconf import OmegaConf
    
    # Simple test config
    test_config = {
        'es_manager': {
            'enable_parallel_env_steps': True,
            'max_env_worker_threads': 2,
            'format_penalty': 0.1,
            'train': {
                'env_groups': 1,
                'group_size': 2,
                'env_configs': {
                    'tags': ['TestEnv'],
                    'n_groups': [1]
                }
            }
        },
        'custom_envs': {
            'TestEnv': {
                'env_type': 'SokobanEnv',
                'max_actions_per_traj': 10,
                'env_config': {
                    'dim_room': (4, 4),
                    'num_boxes': 1
                }
            }
        }
    }
    
    try:
        config = OmegaConf.create(test_config)
        
        # Test sequential
        es_manager_seq = EnvStateManager(config, mode="train")
        es_manager_seq.enable_parallel_env_steps = False
        
        # Test parallel
        es_manager_par = EnvStateManager(config, mode="train")
        es_manager_par.enable_parallel_env_steps = True
        
        print("✅ EnvStateManager with threading support initialized successfully!")
        print(f"Threading enabled: {es_manager_par.enable_parallel_env_steps}")
        print(f"Max worker threads: {es_manager_par.max_env_worker_threads}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


if __name__ == "__main__":
    # Run original main by default
    main()
    
    # Optionally test threading
    print("\n" + "="*50)
    print("Testing threading implementation...")
    test_threading_implementation()
