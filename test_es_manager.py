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
                    'max_steps': 100,
                    'search_depth': 300,
                    'render_mode': 'text'
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True  # Global parallel flag
    })

    try:
        es_manager = EnvStateManager(config, mode='train')
        print('✓ EnvStateManager initialized successfully with parallel execution')
        print(f'✓ Using parallel execution: {es_manager.use_parallel}')
        print(f'✓ Environment type: {es_manager.env_type if es_manager.use_parallel else "mixed"}')
        print(f'✓ Total environments: {len(es_manager.envs)}')
        
        # Test reset
        rollout_cache = es_manager.reset(seed=123)
        print(f'✓ Reset successful, got {len(rollout_cache)} environments')
        
        # Verify rollout cache structure
        if rollout_cache:
            cache_entry = rollout_cache[0]
            expected_keys = {'env_id', 'history', 'group_id', 'tag', 'penalty'}
            assert all(key in cache_entry for key in expected_keys), f"Missing cache keys: {expected_keys - set(cache_entry.keys())}"
            print('✓ Rollout cache structure is correct')
        
        # Test render
        renders = es_manager.render()
        print(f'✓ Render successful, got {len(renders)} renders')
        print(f'✓ First render preview: {renders[0][:100]}...')
        
        # Test a simple step with single action
        all_env_inputs = [
            {
                "env_id": 0,
                "llm_raw_response": "<action>down</action>",
                "llm_response": "<action>down</action>",
                "actions": ["down"]
            }
        ]
        env_outputs = es_manager.step(all_env_inputs)
        print(f'✓ Single action step successful, got {len(env_outputs)} active environments')
        
        # Test step with multiple actions
        all_env_inputs = [
            {
                "env_id": 1,
                "llm_raw_response": "<action>up</action>, <action>left</action>",
                "llm_response": "<action>up</action>, <action>left</action>",
                "actions": ["up", "left"]
            }
        ]
        env_outputs = es_manager.step(all_env_inputs)
        print(f'✓ Multi-action step successful, got {len(env_outputs)} active environments')
        
        # Test invalid action handling
        all_env_inputs = [
            {
                "env_id": 2,
                "llm_raw_response": "invalid action format",
                "llm_response": "invalid action format", 
                "actions": ["invalid_move"]
            }
        ]
        env_outputs = es_manager.step(all_env_inputs)
        print(f'✓ Invalid action handling successful, got {len(env_outputs)} active environments')
        
        # Test getting final rollout states
        final_states = es_manager.get_rollout_states()
        print(f'✓ Get rollout states successful, got {len(final_states)} final states')
        
        # Clean up
        es_manager.close()
        print('✓ Cleanup successful')
        print('✓ All parallel execution tests passed!')
        
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
                    'max_steps': 100,
                    'search_depth': 300,
                    'render_mode': 'text'
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': False  # Global parallel flag disabled
    })

    try:
        es_manager = EnvStateManager(config, mode='train')
        print('✓ EnvStateManager initialized successfully with single-process execution')
        print(f'✓ Using parallel execution: {es_manager.use_parallel}')
        print(f'✓ Total environments: {len(es_manager.envs)}')
        
        # Test reset
        rollout_cache = es_manager.reset(seed=123)
        print(f'✓ Reset successful, got {len(rollout_cache)} environments')
        
        # Test step to ensure single-process mode works
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
        print('✓ Single-process fallback test passed!')
        
    except Exception as e:
        print(f'✗ Error: {e}')
        import traceback
        traceback.print_exc()
        return False
    
    return True

def test_configuration_edge_cases():
    print('\nTesting configuration edge cases...')
    
    # Test with missing parallel availability
    config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'train': {
                'env_groups': 1,
                'group_size': 1,
                'env_configs': {
                    'tags': ['SimpleSokoban'],
                    'n_groups': [1]
                }
            }
        },
        'custom_envs': {
            'SimpleSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 10,
                'env_config': {
                    'dim_room': [4, 4],
                    'num_boxes': 1,
                    'max_steps': 20
                }
            }
        },
        'seed': 42
    })
    
    try:
        es_manager = EnvStateManager(config, mode='train')
        print(f'✓ Edge case config successful, using parallel: {es_manager.use_parallel}')
        es_manager.close()
        print('✓ Edge case test passed!')
        return True
    except Exception as e:
        print(f'✗ Edge case error: {e}')
        return False

if __name__ == "__main__":
    success1 = test_parallel_execution()
    success2 = test_single_process_fallback()
    success3 = test_configuration_edge_cases()
    
    if success1 and success2 and success3:
        print('\n🎉 All tests passed! EnvStateManager integration successful.')
    else:
        print('\n❌ Some tests failed.')
        sys.exit(1) 