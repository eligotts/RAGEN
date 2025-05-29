#!/usr/bin/env python3
"""
Simple test to validate environment configuration fixes
"""
import sys
sys.path.append('.')

from ragen.llm_agent.es_manager import EnvStateManager
from omegaconf import DictConfig

def test_sokoban():
    """Test sokoban with parallel execution"""
    print("Testing Sokoban...")
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {
                    'tags': ['TestSokoban'],
                    'n_groups': [1],
                }
            }
        },
        'custom_envs': {
            'TestSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 5,
                'env_config': {
                    'dim_room': [6, 6],
                    'num_boxes': 1,
                    'max_steps': 50,
                    'render_mode': 'text'
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True
    })
    
    try:
        manager = EnvStateManager(config, mode='train')
        manager.reset(seed=123)
        print("✓ Sokoban parallel execution works")
        manager.close()
        return True
    except Exception as e:
        print(f"❌ Sokoban failed: {e}")
        return False

def test_bandit():
    """Test bandit with parallel execution"""
    print("Testing Bandit...")
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {
                    'tags': ['TestBandit'],
                    'n_groups': [1],
                }
            }
        },
        'custom_envs': {
            'TestBandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': {
                    'lo_arm_name': 'phoenix',
                    'hi_arm_name': 'dragon',
                    'action_space_start': 1,
                    'lo_arm_score': 0.1,
                    'hi_arm_hiscore': 1.0,
                    'hi_arm_loscore': 0.0,
                    'hi_arm_hiscore_prob': 0.25,
                    'render_mode': 'text'
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True
    })
    
    try:
        manager = EnvStateManager(config, mode='train')
        manager.reset(seed=123)
        print("✓ Bandit parallel execution works")
        manager.close()
        return True
    except Exception as e:
        print(f"❌ Bandit failed: {e}")
        return False

def test_frozen_lake():
    """Test frozen lake with parallel execution"""
    print("Testing FrozenLake...")
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {
                    'tags': ['TestFrozenLake'],
                    'n_groups': [1],
                }
            }
        },
        'custom_envs': {
            'TestFrozenLake': {
                'env_type': 'frozen_lake',
                'max_actions_per_traj': 5,
                'env_config': {
                    'size': 4,
                    'p': 0.8,
                    'is_slippery': False,
                    'map_seed': 42,
                    'render_mode': 'text'
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True
    })
    
    try:
        manager = EnvStateManager(config, mode='train')
        manager.reset(seed=123)
        print("✓ FrozenLake parallel execution works")
        manager.close()
        return True
    except Exception as e:
        print(f"❌ FrozenLake failed: {e}")
        return False

if __name__ == "__main__":
    print("Testing environment configuration fixes...")
    
    all_passed = True
    all_passed &= test_sokoban()
    all_passed &= test_bandit()
    all_passed &= test_frozen_lake()
    
    if all_passed:
        print("\n🎉 All configuration tests passed!")
    else:
        print("\n❌ Some configuration tests failed") 