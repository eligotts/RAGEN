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


def worker_func(remote: mp.connection.Connection, seed: int, env_class, config_class, env_kwargs: dict):
    """
    Core subprocess loop that runs environment simulation in isolation.
    
    Args:
        remote: Communication pipe to parent process
        seed: Random seed for environment
        env_class: Environment class (pre-resolved for performance)
        config_class: Configuration class (pre-resolved for performance)
        env_kwargs: Environment configuration parameters
    """
    env = None  # Initialize env to None for the finally block
    try:
        # --- Initialization Phase ---
        # Create environment config (pre-validated in main process)
        env_config = config_class(**env_kwargs) if env_kwargs else config_class()
        
        # Create environment instance (classes pre-resolved for performance)
        env = env_class(env_config)
        
        # Signal readiness to main process
        remote.send(('ready', None))
        
    except Exception as setup_exception:
        # --- Error during setup ONLY ---
        error_msg = f"Worker setup error: {str(setup_exception)}\n{traceback.format_exc()}"
        try:
            remote.send(('init_error', error_msg))
        except Exception:  # If sending the error itself fails
            pass
        # Close environment if partially initialized
        if env is not None:
            try:
                env.close()
            except:
                pass
        try:
            remote.close()
        except:
            pass
        return  # Exit worker process

    current_command = "None"  # For better error reporting
    try:
        # --- Operational Phase ---
        # Main communication loop
        while True:
            cmd, data = remote.recv()  # Block waiting for command
            current_command = cmd
            
            if cmd == 'step':
                # Execute single action in environment
                obs, reward, done, info = env.step(data)
                remote.send((obs, reward, done, info))
                
            elif cmd == 'reset':
                # Reset environment to initial state
                reset_seed = data if data is not None else seed
                # Standard gym-like envs return obs, info. Adapt if your envs differ.
                reset_result = env.reset(seed=reset_seed)
                if isinstance(reset_result, tuple) and len(reset_result) == 2:
                    obs, info = reset_result
                else:  # Assuming env.reset() might just return obs for some envs
                    obs = reset_result
                    info = getattr(env, '_last_info', {})  # Fallback if info not directly returned
                remote.send((obs, info))
                
            elif cmd == 'render':
                # Render current environment state
                mode = data if data is not None else 'text'
                try:
                    # First try with mode parameter (for environments like Sokoban)
                    rendered = env.render(mode=mode)
                except TypeError:
                    try:
                        # Fall back to no parameters (for environments like Bandit, FrozenLake, MetaMathQA)
                        rendered = env.render()
                    except Exception as e_render:
                        raise RuntimeError(f"Render failed for env {env_class.__name__} even without mode: {e_render}") from e_render
                remote.send(rendered)
                
            elif cmd == 'get_actions':
                # Get available actions (for discrete action envs)
                if hasattr(env, 'get_all_actions'):
                    actions = env.get_all_actions()
                else:
                    actions = []
                remote.send(actions)
                
            elif cmd == 'get_action_lookup':
                # Get action lookup mapping (for bandit envs)
                if hasattr(env, 'ACTION_LOOKUP'):
                    action_lookup = env.ACTION_LOOKUP
                elif hasattr(env.config, 'action_lookup'):  # Check config too
                    action_lookup = env.config.action_lookup
                else:
                    action_lookup = {}
                remote.send(action_lookup)
                
            elif cmd == 'ping':
                # Readiness check
                remote.send('ready')
                
            elif cmd == 'close':
                break  # Will go to finally block
                
            else:
                # This case should ideally not be hit if parent sends valid commands
                # If it's an unknown command, it's an operational error.
                raise NotImplementedError(f"Unknown command received by worker: {cmd}")
                
    except EOFError:
        # This is usually normal: parent closed the connection.
        pass
    except Exception as operational_exception:
        # --- Error during operation (step, reset, render AFTER 'ready') ---
        # Letting the worker crash is often the cleanest for debugging.
        # The parent will detect the dead pipe.
        # Print error for debugging
        print(f"Worker operational error (command: {current_command}): {str(operational_exception)}\n{traceback.format_exc()}", file=sys.stderr)
        # Re-raise to ensure the process exits with an error status
        raise
    finally:
        # Cleanup
        if env is not None:
            try:
                env.close()
            except:
                pass
        # Ensure remote is closed if it's not already by 'close' command or EOF
        if hasattr(remote, 'close') and not getattr(remote, 'closed', True):
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
        
        # Pre-resolve environment classes for performance (avoid repeated lookups)
        if env_type not in REGISTERED_ENVS:
            raise ValueError(f"Unknown environment type: {env_type}")
        env_class = REGISTERED_ENVS[env_type]
        config_class = REGISTERED_ENV_CONFIGS[env_type]
        
        # Validate configuration in main process (catch errors early)
        try:
            if self.env_kwargs:
                test_config = config_class(**self.env_kwargs)
            else:
                test_config = config_class()
        except Exception as config_error:
            raise ValueError(f"Failed to create config for {env_type}: {config_error}. "
                           f"Expected params: {getattr(config_class, '__dataclass_fields__', {}).keys()}, "
                           f"Got: {list(self.env_kwargs.keys())}")
        
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
                args=(child_remote, worker_seed, env_class, config_class, self.env_kwargs)
            )
            worker.daemon = True  # Auto-cleanup on main process exit
            worker.start()
            
            child_remote.close()  # Close child end in parent process
            self.parent_remotes.append(parent_remote)
            self.workers.append(worker)
        
        # Wait for all workers to be ready (move initialization cost here)
        # NOTE: This represents the true one-time cost of parallel execution setup.
        # By synchronizing here, we ensure reset() is fast (called many times during training)
        # at the cost of slower initialization (called once per training run).
        # This follows best practices: front-load one-time costs, optimize frequent operations.
        ready_count = 0
        for i, remote in enumerate(self.parent_remotes):
            try:
                status, _ = remote.recv()  # Wait for readiness signal
                if status == 'ready':
                    ready_count += 1
                elif status == 'init_error':
                    raise RuntimeError(f"Worker {i} failed to initialize: {_}")
            except Exception as e:
                raise RuntimeError(f"Worker {i} initialization error: {e}")
        
        print(f"Created {self.num_processes} environment processes for {env_type} (all ready)")
        
        if ready_count != self.num_processes:
            raise RuntimeError(f"Only {ready_count}/{self.num_processes} workers ready")
    
    def step(self, actions: List[Any]) -> Tuple[List, List, List, List]:
        """
        Execute actions across all environments in parallel.
        
        Args:
            actions: List of actions, one per environment process (None means no action)
            
        Returns:
            observations, rewards, dones, infos (all as lists)
        """
        assert len(actions) == self.num_processes, \
            f"Expected {self.num_processes} actions, got {len(actions)}"
        
        # PHASE 1: Send all actions in parallel
        for i, remote in enumerate(self.parent_remotes):
            if actions[i] is not None:  # Only send action if it's not None
                remote.send(('step', actions[i]))
            else:
                # For None actions, send a no-op command that returns current state
                remote.send(('render', 'text'))
        
        # PHASE 2: Collect all results
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for i, remote in enumerate(self.parent_remotes):
            if actions[i] is not None:
                # Real step result
                obs, reward, done, info = remote.recv()
                obs_list.append(obs)
                reward_list.append(reward)
                done_list.append(done)
                info_list.append(info)
            else:
                # No-op result (just render)
                obs = remote.recv()
                obs_list.append(obs)
                reward_list.append(0.0)  # No reward for no action
                done_list.append(False)  # No state change
                info_list.append({})     # No info
        
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
        for remote, seed in zip(self.parent_remotes, seeds):
            remote.send(('reset', seed))
        
        # Collect results
        obs_list, info_list = [], []
        for remote in self.parent_remotes:
            obs, info = remote.recv()
            obs_list.append(obs)
            info_list.append(info)
        
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
        for remote in self.parent_remotes:
            result = remote.recv()
            rendered_list.append(result)
        
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
        for remote in self.parent_remotes:
            result = remote.recv()
            actions_list.append(result)
        
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