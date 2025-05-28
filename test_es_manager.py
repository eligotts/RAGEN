#!/usr/bin/env python3
"""
Test script for EnvStateManager with parallel execution
"""
import sys
sys.path.append('.')

from ragen.llm_agent.es_manager import EnvStateManager
from omegaconf import DictConfig

def test_parallel_execution():
    print('Testing EnvStateManager with parallel execution...')

    # Create a minimal config for testing
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'use_parallel_execution': True,
            'train': {
                'env_groups': 2,
                'group_size': 2,
                'env_configs': {
                    'tags': ['SimpleSokoban'],
                    'n_groups': [2]
                }
            }
        },
        'custom_envs': {
            'SimpleSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 50,
                'env_config': {
                    'dim_room': [6, 6],
                    'num_boxes': 1,
                    'max_steps': 100
                }
            }
        },
        'seed': 42
    })

    try:
        es_manager = EnvStateManager(config, mode='train')
        print('✓ EnvStateManager initialized successfully with parallel execution')
        print(f'✓ Using parallel execution: {es_manager.use_parallel}')
        
        # Test reset
        rollout_cache = es_manager.reset(seed=123)
        print(f'✓ Reset successful, got {len(rollout_cache)} environments')
        
        # Test render
        renders = es_manager.render()
        print(f'✓ Render successful, got {len(renders)} renders')
        print(f'✓ First render preview: {renders[0][:100]}...')
        
        # Test a simple step
        all_env_inputs = [
            {
                "env_id": 0,
                "llm_raw_response": "<action>down</action>",
                "llm_response": "<action>down</action>",
                "actions": ["down"]
            }
        ]
        env_outputs = es_manager.step(all_env_inputs)
        print(f'✓ Step successful, got {len(env_outputs)} active environments')
        
        # Clean up
        es_manager.close()
        print('✓ Cleanup successful')
        print('✓ All tests passed!')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_single_process_fallback():
    print('\nTesting EnvStateManager with single-process fallback...')

    # Create a config with parallel execution disabled
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'use_parallel_execution': False,
            'train': {
                'env_groups': 2,
                'group_size': 2,
                'env_configs': {
                    'tags': ['SimpleSokoban'],
                    'n_groups': [2]
                }
            }
        },
        'custom_envs': {
            'SimpleSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 50,
                'env_config': {
                    'dim_room': [6, 6],
                    'num_boxes': 1,
                    'max_steps': 100
                }
            }
        },
        'seed': 42
    })

    try:
        es_manager = EnvStateManager(config, mode='train')
        print('✓ EnvStateManager initialized successfully with single-process execution')
        print(f'✓ Using parallel execution: {es_manager.use_parallel}')
        
        # Test reset
        rollout_cache = es_manager.reset(seed=123)
        print(f'✓ Reset successful, got {len(rollout_cache)} environments')
        
        # Clean up
        es_manager.close()
        print('✓ Cleanup successful')
        print('✓ Single-process fallback test passed!')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success1 = test_parallel_execution()
    success2 = test_single_process_fallback()
    
    if success1 and success2:
        print('\n🎉 All tests passed! EnvStateManager integration successful.')
    else:
        print('\n❌ Some tests failed.')
        sys.exit(1) 