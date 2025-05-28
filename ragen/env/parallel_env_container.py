"""
Multi-Process Environment Container for parallel environment execution.
This module provides infrastructure for running multiple environment instances
in separate subprocesses to enable true parallelism.
"""

import multiprocessing as mp
import numpy as np
import torch
import sys
import traceback
from typing import List, Tuple, Dict, Any, Optional, Callable
from ragen.env import REGISTERED_ENVS, REGISTERED_ENV_CONFIGS


def worker_func(remote: mp.connection.Connection, seed: int, env_type: str, env_kwargs: dict):
    """
    Core subprocess loop that runs environment simulation in isolation.
    
    Args:
        remote: Communication pipe to parent process
        seed: Random seed for environment
        env_type: Type of environment to create
        env_kwargs: Environment configuration parameters
    """
    try:
        # Initialize environment instance in subprocess
        if env_type not in REGISTERED_ENVS:
            raise ValueError(f"Unknown environment type: {env_type}")
        
        env_class = REGISTERED_ENVS[env_type]
        config_class = REGISTERED_ENV_CONFIGS[env_type]
        
        # Create environment config
        if env_kwargs:
            env_config = config_class(**env_kwargs)
        else:
            env_config = config_class()
        
        # Create environment instance
        env = env_class(env_config)
        
        # Main communication loop
        while True:
            try:
                cmd, data = remote.recv()  # Block waiting for command
                
                if cmd == 'step':
                    # Execute single action in environment
                    obs, reward, done, info = env.step(data)
                    remote.send(('success', (obs, reward, done, info)))
                    
                elif cmd == 'reset':
                    # Reset environment to initial state
                    reset_seed = data if data is not None else seed
                    obs = env.reset(seed=reset_seed)
                    # Create dummy info dict if not returned by reset
                    info = getattr(env, '_last_info', {})
                    remote.send(('success', (obs, info)))
                    
                elif cmd == 'render':
                    # Render current environment state
                    mode = data if data is not None else 'text'
                    rendered = env.render(mode=mode)
                    remote.send(('success', rendered))
                    
                elif cmd == 'get_actions':
                    # Get available actions (for discrete action envs)
                    if hasattr(env, 'get_all_actions'):
                        actions = env.get_all_actions()
                    else:
                        actions = []
                    remote.send(('success', actions))
                    
                elif cmd == 'get_action_lookup':
                    # Get action lookup mapping (for bandit envs)
                    if hasattr(env, 'ACTION_LOOKUP'):
                        action_lookup = env.ACTION_LOOKUP
                    else:
                        action_lookup = {}
                    remote.send(('success', action_lookup))
                    
                elif cmd == 'close':
                    remote.close()
                    break
                    
                else:
                    remote.send(('error', f"Unknown command: {cmd}"))
                    
            except Exception as step_error:
                # Send error but continue running
                error_msg = f"Step error: {str(step_error)}\n{traceback.format_exc()}"
                remote.send(('error', error_msg))
                
    except Exception as init_error:
        # Fatal error during initialization
        error_msg = f"Init error: {str(init_error)}\n{traceback.format_exc()}"
        remote.send(('fatal_error', error_msg))
    finally:
        # Cleanup
        if 'env' in locals():
            try:
                env.close()
            except:
                pass
        try:
            remote.close()
        except:
            pass


class MultiProcessEnvironmentContainer:
    """
    Manages multiple environment instances running in separate subprocesses.
    Enables true parallel execution across CPU processes while maintaining
    efficient coordination.
    """
    
    def __init__(self, env_type: str, env_num: int, group_n: int, seed: int, 
                 env_kwargs: Optional[dict] = None):
        """
        Initialize multi-process environment container.
        
        Args:
            env_type: Type of environment (must be in REGISTERED_ENVS)
            env_num: Number of unique environment configurations
            group_n: Number of environments per group (for GRPO/GiGPO algorithms)
            seed: Base random seed
            env_kwargs: Environment-specific configuration
        """
        self.env_type = env_type
        self.env_num = env_num
        self.group_n = group_n
        self.num_processes = env_num * group_n  # Total parallel processes
        self.env_kwargs = env_kwargs or {}
        
        # Communication infrastructure
        self.parent_remotes: List[mp.connection.Connection] = []
        self.workers: List[mp.Process] = []
        
        # Create subprocess for each environment
        ctx = mp.get_context('spawn')  # Use 'spawn' for cross-platform compatibility
        
        for i in range(self.num_processes):
            parent_remote, child_remote = mp.Pipe()
            
            # Calculate seed: environments in same group share initial seed
            worker_seed = seed + (i // self.group_n)
            
            worker = ctx.Process(
                target=worker_func,
                args=(child_remote, worker_seed, env_type, self.env_kwargs)
            )
            worker.daemon = True  # Auto-cleanup on main process exit
            worker.start()
            
            child_remote.close()  # Close child end in parent process
            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)
        
        print(f"Created {self.num_processes} environment processes for {env_type}")
    
    def step(self, actions: List[Any]) -> Tuple[List, List, List, List]:
        """
        Execute actions across all environments in parallel.
        
        Args:
            actions: List of actions, one per environment process
            
        Returns:
            observations, rewards, dones, infos (all as lists)
        """
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"
        
        # PHASE 1: Send all actions (non-blocking)
        for i, remote in enumerate(self.parent_remotes):
            try:
                remote.send(('step', actions[i]))
            except Exception as e:
                print(f"Error sending action to process {i}: {e}")
                raise
        
        # PHASE 2: Collect all results (blocking)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self.parent_remotes):
            try:
                status, result = remote.recv()
                if status == 'success':
                    obs, reward, done, info = result
                    obs_list.append(obs)
                    reward_list.append(reward)
                    done_list.append(done)
                    info_list.append(info)
                elif status == 'error':
                    print(f"Environment {i} error: {result}")
                    # Use default values for failed environment
                    obs_list.append("")
                    reward_list.append(0.0)
                    done_list.append(True)
                    info_list.append({'error': result})
                else:
                    raise RuntimeError(f"Unexpected status from process {i}: {status}")
            except Exception as e:
                print(f"Error receiving from process {i}: {e}")
                # Use default values for failed environment
                obs_list.append("")
                reward_list.append(0.0)
                done_list.append(True)
                info_list.append({'error': str(e)})
        
        return obs_list, reward_list, done_list, info_list
    
    def reset(self, seeds: Optional[List[int]] = None) -> Tuple[List, List]:
        """
        Reset all environments simultaneously.
        
        Args:
            seeds: Optional list of seeds for each environment
            
        Returns:
            observations, infos (both as lists)
        """
        if seeds is None:
            seeds = [None] * self.num_processes
        else:
            assert len(seeds) == self.num_processes
        
        # Send reset commands
        for i, remote in enumerate(self.parent_remotes):
            try:
                remote.send(('reset', seeds[i]))
            except Exception as e:
                print(f"Error sending reset to process {i}: {e}")
                raise
        
        # Collect results
        obs_list, info_list = [], []
        for i, remote in enumerate(self.parent_remotes):
            try:
                status, result = remote.recv()
                if status == 'success':
                    obs, info = result
                    obs_list.append(obs)
                    info_list.append(info)
                elif status == 'error':
                    print(f"Environment {i} reset error: {result}")
                    obs_list.append("")
                    info_list.append({'error': result})
                else:
                    raise RuntimeError(f"Unexpected status from process {i}: {status}")
            except Exception as e:
                print(f"Error receiving reset from process {i}: {e}")
                obs_list.append("")
                info_list.append({'error': str(e)})
        
        return obs_list, info_list
    
    def render(self, mode: str = 'text') -> List[Any]:
        """
        Render all environments.
        
        Args:
            mode: Rendering mode
            
        Returns:
            List of rendered observations
        """
        # Send render commands
        for remote in self.parent_remotes:
            remote.send(('render', mode))
        
        # Collect results
        rendered_list = []
        for i, remote in enumerate(self.parent_remotes):
            try:
                status, result = remote.recv()
                if status == 'success':
                    rendered_list.append(result)
                else:
                    print(f"Environment {i} render error: {result}")
                    rendered_list.append("")
            except Exception as e:
                print(f"Error receiving render from process {i}: {e}")
                rendered_list.append("")
        
        return rendered_list
    
    def get_available_actions(self) -> List[List]:
        """
        Get available actions for all environments.
        
        Returns:
            List of action lists for each environment
        """
        # Send get_actions commands
        for remote in self.parent_remotes:
            remote.send(('get_actions', None))
        
        # Collect results
        actions_list = []
        for i, remote in enumerate(self.parent_remotes):
            try:
                status, result = remote.recv()
                if status == 'success':
                    actions_list.append(result)
                else:
                    print(f"Environment {i} get_actions error: {result}")
                    actions_list.append([])
            except Exception as e:
                print(f"Error receiving actions from process {i}: {e}")
                actions_list.append([])
        
        return actions_list
    
    def close(self):
        """
        Close all subprocesses and clean up resources.
        """
        # Send close commands
        for remote in self.parent_remotes:
            try:
                remote.send(('close', None))
            except:
                pass  # Remote might already be closed
        
        # Wait for processes to terminate
        for worker in self.workers:
            worker.join(timeout=5.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=2.0)
                if worker.is_alive():
                    worker.kill()  # Force kill as last resort
        
        # Close remotes
        for remote in self.parent_remotes:
            try:
                remote.close()
            except:
                pass
        
        self.parent_remotes.clear()
        self.workers.clear()
    
    def __del__(self):
        """Destructor to ensure cleanup."""
        self.close()


def to_numpy(data):
    """Convert various data types to numpy arrays."""
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, np.ndarray):
        return data
    elif isinstance(data, (int, float, bool, tuple, list)):
        return np.array(data)
    else:
        raise ValueError(f"Unsupported type: {type(data)}") 