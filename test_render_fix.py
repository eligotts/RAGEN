#!/usr/bin/env python3
"""
Test script to validate render method fixes
"""
import sys
sys.path.append('.')

from ragen.llm_agent.es_manager import EnvStateManager
from omegaconf import DictConfig

def test_bandit_render():
    """Test bandit render (doesn't accept mode)"""
    print("Testing Bandit render...")
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
        
        # Test render
        renders = manager.render()
        print(f"✓ Bandit render successful: {len(renders)} renders")
        
        # Test step with render fallback for None actions
        step_result = manager.step([
            {"env_id": 0, "llm_raw_response": "phoenix", "llm_response": "phoenix", "actions": ["phoenix"]}
        ])
        print(f"✓ Bandit step successful: {len(step_result)} active envs")
        
        manager.close()
        return True
    except Exception as e:
        print(f"❌ Bandit render test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_sokoban_render():
    """Test sokoban render (accepts mode)"""
    print("Testing Sokoban render...")
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
        
        # Test render
        renders = manager.render()
        print(f"✓ Sokoban render successful: {len(renders)} renders")
        
        manager.close()
        return True
    except Exception as e:
        print(f"❌ Sokoban render test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("Testing render method fixes...")
    
    all_passed = True
    all_passed &= test_sokoban_render()
    all_passed &= test_bandit_render()
    
    if all_passed:
        print("\n🎉 All render tests passed!")
    else:
        print("\n❌ Some render tests failed") 