"""
Environment Factory for creating and configuring parallel environments.
This module provides centralized environment instantiation with proper configuration
for different environment types.
"""

from typing import Tuple, Dict, Any, Optional
from .parallel_env_container import MultiProcessEnvironmentContainer
from .environment_manager import (
    EnvironmentManagerBase, SokobanEnvironmentManager, WebShopEnvironmentManager,
    CountdownEnvironmentManager, MetaMathQAEnvironmentManager, 
    FrozenLakeEnvironmentManager, BanditEnvironmentManager
)
from .projection_functions import create_projection_function


def make_environments(config) -> Tuple[EnvironmentManagerBase, EnvironmentManagerBase]:
    """
    Create training and validation environments based on configuration.
    
    Args:
        config: Training configuration object
        
    Returns:
        train_envs, val_envs: Environment managers for training and validation
    """
    # Extract environment configuration
    env_configs = config.es_manager.train.env_configs
    if not env_configs.tags:
        raise ValueError("No environment tags specified in config")
    
    # For now, support single environment type
    # TODO: Add support for mixed environment types
    env_tag = env_configs.tags[0]
    env_type = config.custom_envs[env_tag].env_type
    
    # Calculate environment counts
    train_env_num = config.es_manager.train.env_groups
    train_group_n = config.es_manager.train.group_size
    val_env_num = config.es_manager.val.env_groups  
    val_group_n = config.es_manager.val.group_size
    
    # Create environment kwargs from config
    env_kwargs = _extract_env_kwargs(config, env_tag)
    
    # Create training environment container
    train_container = MultiProcessEnvironmentContainer(
        env_type=env_type,
        env_num=train_env_num,
        group_n=train_group_n,
        seed=config.get('seed', 42),
        env_kwargs=env_kwargs
    )
    
    # Create validation environment container
    val_container = MultiProcessEnvironmentContainer(
        env_type=env_type,
        env_num=val_env_num,
        group_n=val_group_n,
        seed=config.get('seed', 42) + 1000,  # Different seed for validation
        env_kwargs=env_kwargs
    )
    
    # Create projection function
    projection_f = create_projection_function(env_type)
    
    # Create environment managers
    train_envs = _create_environment_manager(
        env_type, train_container, projection_f, env_tag
    )
    val_envs = _create_environment_manager(
        env_type, val_container, projection_f, env_tag
    )
    
    print(f"Created parallel environments for {env_type}")
    print(f"Training: {train_env_num} groups × {train_group_n} envs = {train_env_num * train_group_n} total")
    print(f"Validation: {val_env_num} groups × {val_group_n} envs = {val_env_num * val_group_n} total")
    
    return train_envs, val_envs


def _extract_env_kwargs(config, env_tag: str) -> Dict[str, Any]:
    """
    Extract environment-specific configuration parameters.
    
    Args:
        config: Full configuration object
        env_tag: Environment tag to extract config for
        
    Returns:
        Dictionary of environment kwargs
    """
    env_config = config.custom_envs[env_tag]
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
            # Add WebShop-specific parameters
            for key, value in webshop_config.items():
                env_kwargs[key] = value
    
    elif env_type == 'countdown':
        if hasattr(env_config, 'env_config') and env_config.env_config:
            countdown_config = env_config.env_config
            # Add Countdown-specific parameters
            for key, value in countdown_config.items():
                env_kwargs[key] = value
    
    elif env_type == 'metamathqa':
        if hasattr(env_config, 'env_config') and env_config.env_config:
            metamath_config = env_config.env_config
            # Add MetaMathQA-specific parameters
            for key, value in metamath_config.items():
                env_kwargs[key] = value
    
    elif env_type == 'frozen_lake':
        if hasattr(env_config, 'env_config') and env_config.env_config:
            frozen_lake_config = env_config.env_config
            # Add FrozenLake-specific parameters
            for key, value in frozen_lake_config.items():
                env_kwargs[key] = value
    
    elif env_type == 'bandit':
        if hasattr(env_config, 'env_config') and env_config.env_config:
            bandit_config = env_config.env_config
            # Add Bandit-specific parameters
            for key, value in bandit_config.items():
                env_kwargs[key] = value
    
    return env_kwargs


def _create_environment_manager(env_type: str, container: MultiProcessEnvironmentContainer,
                               projection_f, env_name: str) -> EnvironmentManagerBase:
    """
    Create appropriate environment manager for the given environment type.
    
    Args:
        env_type: Type of environment
        container: Multi-process environment container
        projection_f: Action projection function
        env_name: Environment name/tag
        
    Returns:
        Appropriate environment manager instance
    """
    if env_type == 'sokoban':
        return SokobanEnvironmentManager(container, projection_f, env_name)
    elif env_type == 'webshop':
        return WebShopEnvironmentManager(container, projection_f, env_name)
    elif env_type == 'countdown':
        return CountdownEnvironmentManager(container, projection_f, env_name)
    elif env_type == 'metamathqa':
        return MetaMathQAEnvironmentManager(container, projection_f, env_name)
    elif env_type == 'frozen_lake':
        return FrozenLakeEnvironmentManager(container, projection_f, env_name)
    elif env_type == 'bandit':
        return BanditEnvironmentManager(container, projection_f, env_name)
    else:
        # Use base manager for unknown types
        return EnvironmentManagerBase(container, projection_f, env_name)


def create_single_environment_for_testing(env_type: str, seed: int = 42, 
                                         env_kwargs: Optional[Dict] = None) -> EnvironmentManagerBase:
    """
    Create a single environment for testing purposes.
    
    Args:
        env_type: Type of environment to create
        seed: Random seed
        env_kwargs: Environment configuration parameters
        
    Returns:
        Environment manager with single environment
    """
    container = MultiProcessEnvironmentContainer(
        env_type=env_type,
        env_num=1,
        group_n=1,
        seed=seed,
        env_kwargs=env_kwargs or {}
    )
    
    projection_f = create_projection_function(env_type)
    
    return _create_environment_manager(env_type, container, projection_f, env_type)


def validate_environment_config(config) -> bool:
    """
    Validate environment configuration for consistency.
    
    Args:
        config: Configuration object to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    # Check required fields
    if not hasattr(config, 'es_manager'):
        raise ValueError("Missing es_manager configuration")
    
    if not hasattr(config.es_manager, 'train') or not hasattr(config.es_manager, 'val'):
        raise ValueError("Missing train/val configuration in es_manager")
    
    # Check environment tags
    train_tags = config.es_manager.train.env_configs.tags
    if not train_tags:
        raise ValueError("No training environment tags specified")
    
    # Check custom environment definitions
    if not hasattr(config, 'custom_envs'):
        raise ValueError("Missing custom_envs configuration")
    
    for tag in train_tags:
        if tag not in config.custom_envs:
            raise ValueError(f"Environment tag '{tag}' not found in custom_envs")
        
        env_config = config.custom_envs[tag]
        if not hasattr(env_config, 'env_type'):
            raise ValueError(f"Missing env_type for environment '{tag}'")
    
    # Check batch size consistency
    train_total = config.es_manager.train.env_groups * config.es_manager.train.group_size
    if hasattr(config, 'data') and hasattr(config.data, 'train_batch_size'):
        expected_batch_size = config.data.train_batch_size
        if train_total != expected_batch_size:
            print(f"Warning: Environment count ({train_total}) != train_batch_size ({expected_batch_size})")
    
    return True


# Utility functions for environment management
def get_supported_environment_types() -> list:
    """Get list of supported environment types."""
    return ['sokoban', 'webshop', 'countdown', 'metamathqa', 'frozen_lake', 'bandit']


def get_environment_requirements(env_type: str) -> Dict[str, Any]:
    """
    Get requirements and default configuration for an environment type.
    
    Args:
        env_type: Environment type
        
    Returns:
        Dictionary with requirements and defaults
    """
    requirements = {
        'sokoban': {
            'required_packages': ['gym-sokoban'],
            'default_config': {
                'dim_room': (6, 6),
                'num_boxes': 1,
                'max_steps': 100,
                'search_depth': 30,
                'render_mode': 'text'
            }
        },
        'webshop': {
            'required_packages': ['webshop'],
            'default_config': {
                'max_steps': 50,
            }
        },
        'countdown': {
            'required_packages': [],
            'default_config': {
                'max_steps': 10,
            }
        },
        'metamathqa': {
            'required_packages': [],
            'default_config': {
                'max_steps': 1,
            }
        },
        'frozen_lake': {
            'required_packages': ['gymnasium'],
            'default_config': {
                'map_name': '4x4',
                'is_slippery': True,
            }
        },
        'bandit': {
            'required_packages': [],
            'default_config': {
                'n_arms': 10,
                'max_steps': 100,
            }
        }
    }
    
    return requirements.get(env_type, {
        'required_packages': [],
        'default_config': {}
    }) 