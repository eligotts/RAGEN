"""
Environment Manager for coordinating parallel environments with LLM training.
This module provides the interface layer between the training loop and 
multi-process environment containers.
"""

import numpy as np
import torch
from typing import List, Dict, Any, Optional, Callable, Tuple
from collections import defaultdict
from .parallel_env_container import MultiProcessEnvironmentContainer, to_numpy


class EnvironmentManagerBase:
    """
    Base class for managing parallel environments.
    Provides standardized interface between training loop and multi-process environments.
    """
    
    def __init__(self, envs: MultiProcessEnvironmentContainer, 
                 projection_f: Callable, env_name: str):
        """
        Initialize environment manager.
        
        Args:
            envs: Multi-process environment container
            projection_f: Function to convert LLM text to environment actions
            env_name: Environment identifier
        """
        self.envs = envs
        self.projection_f = projection_f
        self.env_name = env_name
        
        # History tracking for each environment
        self.buffers: Optional[List[List[Dict]]] = None
        self.previous_obs: Optional[List[str]] = None
        
    def reset(self) -> Tuple[Dict[str, Any], List[Dict]]:
        """
        Reset all environments and return structured observations.
        
        Returns:
            observations: Dict with 'text', 'image', 'anchor' keys
            infos: List of info dicts from environments
        """
        raw_obs, infos = self.envs.reset()
        
        # Initialize history buffers
        if self.buffers is not None:
            self.buffers.clear()
        self.buffers = [[] for _ in range(len(raw_obs))]
        
        # Store previous observations for history tracking
        self.previous_obs = raw_obs.copy()
        
        # Build structured observations
        text_obs = self.build_text_obs(raw_obs, init=True)
        
        observations = {
            'text': text_obs,
            'image': self._extract_images(raw_obs) if self._has_images() else None,
            'anchor': raw_obs  # Raw observations for advanced algorithms
        }
        
        return observations, infos
    
    def step(self, text_actions: List[str]) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, List[Dict]]:
        """
        Execute text actions and return next state.
        
        Args:
            text_actions: Raw text outputs from language model
            
        Returns:
            next_observations: Structured observation dict
            rewards: Reward array
            dones: Done flags array  
            infos: Environment info dicts
        """
        # Convert LLM text to environment actions
        actions, valids = self.projection_f(text_actions, self._get_action_space())
        
        # For bandit environments, convert arm names to action IDs
        if self.env_name == 'bandit':
            converted_actions = []
            for i, action in enumerate(actions):
                if isinstance(action, str):
                    # Get the action lookup from the first environment process
                    # This is a bit hacky but necessary for the bandit environment
                    if hasattr(self.envs, 'parent_remotes') and len(self.envs.parent_remotes) > 0:
                        # Send a request to get action lookup
                        self.envs.parent_remotes[0].send(('get_action_lookup', None))
                        status, action_lookup = self.envs.parent_remotes[0].recv()
                        if status == 'success':
                            # Find the action ID for this arm name
                            action_id = None
                            for aid, arm_name in action_lookup.items():
                                if arm_name.lower() == action.lower():
                                    action_id = aid
                                    break
                            if action_id is not None:
                                converted_actions.append(action_id)
                            else:
                                converted_actions.append(1)  # Default to first action
                        else:
                            converted_actions.append(1)  # Default to first action
                    else:
                        converted_actions.append(1)  # Default to first action
                else:
                    converted_actions.append(action)
            actions = converted_actions
        
        # Execute in parallel environments
        raw_obs, rewards, dones, infos = self.envs.step(actions)
        
        # Save to history for next observation building
        if self.previous_obs is not None:
            self._save_to_history_buffer(self.previous_obs, actions)
        self.previous_obs = raw_obs.copy()
        
        # Add action validity to infos
        for i, info in enumerate(infos):
            info['is_action_valid'] = to_numpy(valids[i])
        
        # Build next observations
        text_obs = self.build_text_obs(raw_obs)
        next_observations = {
            'text': text_obs,
            'image': self._extract_images(raw_obs) if self._has_images() else None,
            'anchor': raw_obs
        }
        
        return next_observations, np.array(rewards), np.array(dones), infos
    
    def build_text_obs(self, raw_obs: List[str], init: bool = False, 
                       history_length: int = 2) -> List[str]:
        """
        Build rich text observations including history and available actions.
        
        Args:
            raw_obs: Raw observations from environments
            init: Whether this is initial observation (no history)
            history_length: Number of previous steps to include
            
        Returns:
            List of formatted text observations for language model
        """
        formatted_obs = []
        
        for i in range(len(raw_obs)):
            if init or history_length <= 0 or not self.buffers or len(self.buffers[i]) == 0:
                # Initial observation without history
                obs = self._format_initial_observation(raw_obs[i], i)
            else:
                # Include action history
                recent_history = self.buffers[i][-history_length:]
                history_text = self._format_history(recent_history)
                obs = self._format_observation_with_history(
                    raw_obs[i], history_text, i
                )
            
            formatted_obs.append(obs)
        
        return formatted_obs
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial observation without history."""
        available_actions = self._get_available_actions_for_env(env_idx)
        return f"""Current Observation: {obs}

Available Actions: {available_actions}

What do you want to do?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format observation with action history context."""
        available_actions = self._get_available_actions_for_env(env_idx)
        
        return f"""Previous Actions:
{history_text}

Current Observation: {current_obs}

Available Actions: {available_actions}

What do you want to do next?"""
    
    def _format_history(self, history: List[Dict]) -> str:
        """Format action history into readable text."""
        history_lines = []
        for i, step in enumerate(history):
            step_num = step.get('step', i + 1)
            obs = step.get('observation', 'N/A')
            action = step.get('action', 'N/A')
            history_lines.append(f"Step {step_num}: Observation: {obs[:100]}... Action: {action}")
        return '\n'.join(history_lines)
    
    def _save_to_history_buffer(self, observations: List[str], actions: List[str]):
        """Save step data to history buffers."""
        if self.buffers is None:
            return
            
        for i in range(min(len(actions), len(observations), len(self.buffers))):
            self.buffers[i].append({
                'observation': observations[i],
                'action': actions[i],
                'step': len(self.buffers[i]) + 1
            })
    
    def _get_action_space(self) -> List[List]:
        """Get available actions for all environments."""
        try:
            return self.envs.get_available_actions()
        except:
            # Fallback if environments don't support action space queries
            return [[] for _ in range(self.envs.num_processes)]
    
    def _get_available_actions_for_env(self, env_idx: int) -> str:
        """Get formatted available actions for a specific environment."""
        try:
            action_spaces = self._get_action_space()
            if env_idx < len(action_spaces) and action_spaces[env_idx]:
                return ', '.join(str(action) for action in action_spaces[env_idx])
            else:
                return "Any valid action"
        except:
            return "Any valid action"
    
    def _extract_images(self, raw_obs: List[Any]) -> Optional[List[Any]]:
        """Extract image data from observations if present."""
        # Override in subclasses for multimodal environments
        return None
    
    def _has_images(self) -> bool:
        """Check if environment provides image observations."""
        # Override in subclasses for multimodal environments
        return False
    
    def close(self) -> None:
        """Close the environment and release resources."""
        self.envs.close()
    
    def success_evaluator(self, *args, **kwargs) -> Dict[str, np.ndarray]:
        """
        Evaluate if the episodes are successful or not.
        Default implementation checks info['success'] of the last step.
        
        Returns:
            success: Dict with success metrics
        """
        total_infos = kwargs.get('total_infos', [])
        total_batch_list = kwargs.get('total_batch_list', [])
        batch_size = len(total_batch_list)
        
        success = defaultdict(list)
        
        for bs in range(batch_size):
            self._process_batch(bs, total_batch_list, total_infos, success)
        
        if len(success['success_rate']) != batch_size:
            # Fallback if no success data found
            success['success_rate'] = [0.0] * batch_size
        
        return {key: np.array(value) for key, value in success.items()}
    
    def _process_batch(self, batch_idx: int, total_batch_list: List[List[Dict]], 
                      total_infos: List[List[Dict]], success: Dict[str, List]):
        """Process a single batch for success evaluation."""
        if batch_idx >= len(total_batch_list):
            success['success_rate'].append(0.0)
            return
            
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item.get('active_masks', True):
                if batch_idx < len(total_infos) and i < len(total_infos[batch_idx]):
                    info = total_infos[batch_idx][i]
                    # Check various success indicators
                    success_value = float(
                        info.get('success', 
                        info.get('won', 
                        info.get('task_success', 0)))
                    )
                    success['success_rate'].append(success_value)
                else:
                    success['success_rate'].append(0.0)
                return
        
        # If no active masks found, assume failure
        success['success_rate'].append(0.0)


class SokobanEnvironmentManager(EnvironmentManagerBase):
    """Environment manager specifically for Sokoban environments."""
    
    ACTION_LOOKUP = {
        1: "Up",
        2: "Down", 
        3: "Left",
        4: "Right",
    }
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial Sokoban observation."""
        return f"""You are playing Sokoban. Push all boxes onto target positions.

Current State:
{obs}

Available Actions: Up, Down, Left, Right

Use the format: <action>direction</action>
Example: <action>up</action>

What is your next move?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format Sokoban observation with history."""
        return f"""You are playing Sokoban. Push all boxes onto target positions.

{history_text}

Current State:
{current_obs}

Available Actions: Up, Down, Left, Right

Use the format: <action>direction</action>
Example: <action>up</action>

What is your next move?"""


class WebShopEnvironmentManager(EnvironmentManagerBase):
    """Environment manager specifically for WebShop environments."""
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial WebShop observation."""
        return f"""You are shopping online. Find and purchase the requested item.

Current Page:
{obs}

You can use actions like:
- search[query] to search for items
- click[button/link] to navigate
- buy to purchase the current item

Use the format: <action>your_action</action>
Example: <action>search[red shoes]</action>

What would you like to do?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format WebShop observation with history."""
        return f"""You are shopping online. Find and purchase the requested item.

{history_text}

Current Page:
{current_obs}

You can use actions like:
- search[query] to search for items  
- click[button/link] to navigate
- buy to purchase the current item

Use the format: <action>your_action</action>
Example: <action>search[red shoes]</action>

What would you like to do next?"""


# Add more environment-specific managers as needed
class CountdownEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for Countdown math environments."""
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial Countdown observation."""
        return f"""You are solving a Countdown math puzzle.

{obs}

Create a mathematical expression using each number exactly once.
Use +, -, *, / operations to reach the target.

Example format: - 1 + 2 + 3

What is your solution?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format Countdown observation with history."""
        return f"""You are solving a Countdown math puzzle.

{history_text}

{current_obs}

Create a mathematical expression using each number exactly once.
Use +, -, *, / operations to reach the target.

Example format: - 1 + 2 + 3

What is your solution?"""


class MetaMathQAEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for MetaMathQA environments."""
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial MetaMathQA observation."""
        return f"""Solve this math problem step by step.

{obs}

Provide your complete solution and final answer."""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format MetaMathQA observation with history."""
        return f"""Solve this math problem step by step.

{history_text}

{current_obs}

Provide your complete solution and final answer."""


class FrozenLakeEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for FrozenLake environments."""
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial FrozenLake observation."""
        return f"""You are navigating a frozen lake. Reach the goal (G) while avoiding holes (O).

Current State:
{obs}

Available Actions: Left, Down, Right, Up

Use the format: <action>direction</action>
Example: <action>right</action>

What is your next move?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format FrozenLake observation with history."""
        return f"""You are navigating a frozen lake. Reach the goal (G) while avoiding holes (O).

{history_text}

Current State:
{current_obs}

Available Actions: Left, Down, Right, Up

Use the format: <action>direction</action>
Example: <action>right</action>

What is your next move?"""


class BanditEnvironmentManager(EnvironmentManagerBase):
    """Environment manager for Bandit environments."""
    
    def _format_initial_observation(self, obs: str, env_idx: int) -> str:
        """Format initial Bandit observation."""
        return f"""{obs}

Analyze the symbolic meaning of each arm's name and choose the one you think will give higher rewards.

Use the format: <answer>arm_name</answer>
Example: <answer>dragon</answer>

Which arm do you choose?"""
    
    def _format_observation_with_history(self, current_obs: str, 
                                       history_text: str, env_idx: int) -> str:
        """Format Bandit observation with history."""
        return f"""{history_text}

{current_obs}

Based on the results, which arm do you choose next?

Use the format: <answer>arm_name</answer>
Example: <answer>dragon</answer>

Which arm do you choose?""" 