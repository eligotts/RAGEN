#!/usr/bin/env python3
"""
Test script for EnvStateManager with parallel execution
"""
import sys
sys.path.append('.')

from ragen.llm_agent.es_manager import EnvStateManager
from omegaconf import DictConfig
import time
import json
import numpy as np

def compare_rollout_cache(cache1, cache2, tolerance=1e-6):
    """Compare two rollout cache entries with detailed logging"""
    differences = []
    
    # Compare basic fields
    for key in ['env_id', 'group_id', 'tag', 'penalty']:
        if cache1.get(key) != cache2.get(key):
            differences.append(f"  {key}: {cache1.get(key)} vs {cache2.get(key)}")
    
    # Compare history lengths
    if len(cache1.get('history', [])) != len(cache2.get('history', [])):
        differences.append(f"  history length: {len(cache1.get('history', []))} vs {len(cache2.get('history', []))}")
        return differences
    
    # Compare history entries
    for i, (h1, h2) in enumerate(zip(cache1.get('history', []), cache2.get('history', []))):
        for key in h1.keys() | h2.keys():
            if key == 'images':
                # Skip image comparison for now (complex)
                continue
            elif key in ['reward', 'actions_left'] and isinstance(h1.get(key), (int, float)) and isinstance(h2.get(key), (int, float)):
                # Numerical comparison with tolerance
                if abs(h1.get(key, 0) - h2.get(key, 0)) > tolerance:
                    differences.append(f"  history[{i}].{key}: {h1.get(key)} vs {h2.get(key)}")
            elif h1.get(key) != h2.get(key):
                differences.append(f"  history[{i}].{key}: {h1.get(key)} vs {h2.get(key)}")
    
    return differences

def test_parallel_vs_single_process():
    """Comprehensive test comparing parallel and single-process execution"""
    print("="*80)
    print("COMPREHENSIVE PARALLEL vs SINGLE-PROCESS TEST")
    print("="*80)
    
    # Create configs for both modes
    base_config = DictConfig({
        'es_manager': {
            'format_penalty': -0.1,
            'train': {
                'env_groups': 2,
                'group_size': 2,
                'env_configs': {
                    'tags': ['SimpleSokoban'],
                    'n_groups': [2],
                }
            }
        },
        'custom_envs': {
            'SimpleSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 10,
                'env_config': {
                    'dim_room': [7, 7],
                    'num_boxes': 1,
                    'max_steps': 200
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True
    })
    
    # Create parallel config
    parallel_config = DictConfig(base_config.copy())
    parallel_config.es_manager.use_parallel_execution = True
    
    # Create single-process config
    single_config = DictConfig(base_config.copy())
    single_config.es_manager.use_parallel_execution = False
    
    print(f"Initializing environments...")
    print(f"  - Parallel execution: {parallel_config.es_manager.use_parallel_execution}")
    print(f"  - Single-process execution: {single_config.es_manager.use_parallel_execution}")
    
    # Initialize both managers
    print("\n" + "-"*50)
    print("INITIALIZATION")
    print("-"*50)
    
    start_time = time.time()
    parallel_manager = EnvStateManager(parallel_config, mode='train')
    parallel_init_time = time.time() - start_time
    print(f"✓ Parallel manager initialized in {parallel_init_time:.4f}s")
    
    start_time = time.time()
    single_manager = EnvStateManager(single_config, mode='train')
    single_init_time = time.time() - start_time
    print(f"✓ Single-process manager initialized in {single_init_time:.4f}s")
    
    # Test 1: Reset comparison
    print("\n" + "-"*50)
    print("TEST 1: RESET COMPARISON")
    print("-"*50)
    
    reset_seed = 12345
    
    start_time = time.time()
    parallel_reset = parallel_manager.reset(seed=reset_seed)
    parallel_reset_time = time.time() - start_time
    print(f"✓ Parallel reset completed in {parallel_reset_time:.4f}s")
    
    start_time = time.time()
    single_reset = single_manager.reset(seed=reset_seed)
    single_reset_time = time.time() - start_time
    print(f"✓ Single-process reset completed in {single_reset_time:.4f}s")
    
    # Compare reset outputs
    print(f"\nReset output comparison:")
    print(f"  Parallel envs: {len(parallel_reset)}")
    print(f"  Single envs: {len(single_reset)}")
    
    reset_differences = []
    for i, (p_cache, s_cache) in enumerate(zip(parallel_reset, single_reset)):
        diffs = compare_rollout_cache(p_cache, s_cache)
        if diffs:
            reset_differences.extend([f"Env {i}:"] + diffs)
    
    if reset_differences:
        print("❌ Reset outputs differ:")
        for diff in reset_differences[:10]:  # Show first 10 differences
            print(f"  {diff}")
        if len(reset_differences) > 10:
            print(f"  ... and {len(reset_differences) - 10} more differences")
    else:
        print("✓ Reset outputs are identical")
    
    # Test 2: Step comparison with multiple steps
    print("\n" + "-"*50)
    print("TEST 2: STEP EXECUTION COMPARISON")
    print("-"*50)
    
    # Define a sequence of test steps
    test_steps = [
        # Step 1: Simple single actions
        [
            {"env_id": 0, "llm_raw_response": "Move down", "llm_response": "Move down", "actions": ["down"]},
            {"env_id": 1, "llm_raw_response": "Move right", "llm_response": "Move right", "actions": ["right"]},
        ],
        # Step 2: Multiple actions
        [
            {"env_id": 0, "llm_raw_response": "Move right, then up", "llm_response": "Move right, then up", "actions": ["right", "up"]},
            {"env_id": 2, "llm_raw_response": "Move left", "llm_response": "Move left", "actions": ["left"]},
        ],
        # Step 3: More complex actions
        [
            {"env_id": 1, "llm_raw_response": "Series of moves", "llm_response": "Series of moves", "actions": ["up", "left", "down"]},
            {"env_id": 3, "llm_raw_response": "Single move", "llm_response": "Single move", "actions": ["right"]},
        ]
    ]
    
    step_times_parallel = []
    step_times_single = []
    
    for step_num, step_inputs in enumerate(test_steps):
        print(f"\nExecuting step {step_num + 1}...")
        print(f"  Actions: {[(inp['env_id'], inp['actions']) for inp in step_inputs]}")
        
        # Execute parallel step
        start_time = time.time()
        parallel_outputs = parallel_manager.step(step_inputs)
        parallel_step_time = time.time() - start_time
        step_times_parallel.append(parallel_step_time)
        
        # Execute single-process step
        start_time = time.time()
        single_outputs = single_manager.step(step_inputs)
        single_step_time = time.time() - start_time
        step_times_single.append(single_step_time)
        
        print(f"  Parallel: {parallel_step_time:.4f}s, outputs: {len(parallel_outputs)}")
        print(f"  Single: {single_step_time:.4f}s, outputs: {len(single_outputs)}")
        
        # Compare step outputs
        if len(parallel_outputs) != len(single_outputs):
            print(f"  ❌ Output count differs: {len(parallel_outputs)} vs {len(single_outputs)}")
            continue
        
        step_differences = []
        for i, (p_out, s_out) in enumerate(zip(parallel_outputs, single_outputs)):
            diffs = compare_rollout_cache(p_out, s_out)
            if diffs:
                step_differences.extend([f"Output {i}:"] + diffs)
        
        if step_differences:
            print(f"  ❌ Step {step_num + 1} outputs differ:")
            for diff in step_differences[:5]:  # Show first 5 differences
                print(f"    {diff}")
            if len(step_differences) > 5:
                print(f"    ... and {len(step_differences) - 5} more differences")
        else:
            print(f"  ✓ Step {step_num + 1} outputs are identical")
    
    # Test 3: Render comparison
    print("\n" + "-"*50)
    print("TEST 3: RENDER COMPARISON")
    print("-"*50)
    
    start_time = time.time()
    parallel_renders = parallel_manager.render()
    parallel_render_time = time.time() - start_time
    
    start_time = time.time()
    single_renders = single_manager.render()
    single_render_time = time.time() - start_time
    
    print(f"Parallel render: {parallel_render_time:.4f}s")
    print(f"Single render: {single_render_time:.4f}s")
    
    if len(parallel_renders) != len(single_renders):
        print(f"❌ Render count differs: {len(parallel_renders)} vs {len(single_renders)}")
    else:
        render_identical = True
        for i, (p_render, s_render) in enumerate(zip(parallel_renders, single_renders)):
            if p_render != s_render:
                print(f"❌ Render {i} differs")
                print(f"  Parallel: {p_render[:100]}...")
                print(f"  Single: {s_render[:100]}...")
                render_identical = False
                break
        
        if render_identical:
            print("✓ All renders are identical")
    
    # Test 4: Final rollout states comparison
    print("\n" + "-"*50)
    print("TEST 4: FINAL ROLLOUT STATES COMPARISON")
    print("-"*50)
    
    start_time = time.time()
    parallel_final = parallel_manager.get_rollout_states()
    parallel_final_time = time.time() - start_time
    
    start_time = time.time()
    single_final = single_manager.get_rollout_states()
    single_final_time = time.time() - start_time
    
    print(f"Parallel final: {parallel_final_time:.4f}s")
    print(f"Single final: {single_final_time:.4f}s")
    
    final_differences = []
    for i, (p_final, s_final) in enumerate(zip(parallel_final, single_final)):
        diffs = compare_rollout_cache(p_final, s_final)
        if diffs:
            final_differences.extend([f"Final env {i}:"] + diffs)
    
    if final_differences:
        print("❌ Final states differ:")
        for diff in final_differences[:10]:
            print(f"  {diff}")
        if len(final_differences) > 10:
            print(f"  ... and {len(final_differences) - 10} more differences")
    else:
        print("✓ Final states are identical")
    
    # Performance summary
    print("\n" + "="*50)
    print("PERFORMANCE SUMMARY")
    print("="*50)
    
    total_parallel_time = parallel_init_time + parallel_reset_time + sum(step_times_parallel) + parallel_render_time + parallel_final_time
    total_single_time = single_init_time + single_reset_time + sum(step_times_single) + single_render_time + single_final_time
    
    print(f"Initialization:")
    print(f"  Parallel: {parallel_init_time:.4f}s")
    print(f"  Single:   {single_init_time:.4f}s")
    print(f"  Speedup:  {single_init_time/parallel_init_time:.2f}x")
    
    print(f"\nReset:")
    print(f"  Parallel: {parallel_reset_time:.4f}s")
    print(f"  Single:   {single_reset_time:.4f}s")
    print(f"  Speedup:  {single_reset_time/parallel_reset_time:.2f}x")
    
    print(f"\nStep execution (average):")
    avg_parallel_step = np.mean(step_times_parallel)
    avg_single_step = np.mean(step_times_single)
    print(f"  Parallel: {avg_parallel_step:.4f}s")
    print(f"  Single:   {avg_single_step:.4f}s")
    print(f"  Speedup:  {avg_single_step/avg_parallel_step:.2f}x")
    
    print(f"\nTotal execution:")
    print(f"  Parallel: {total_parallel_time:.4f}s")
    print(f"  Single:   {total_single_time:.4f}s")
    print(f"  Speedup:  {total_single_time/total_parallel_time:.2f}x")
    
    # Cleanup
    print(f"\nCleaning up...")
    parallel_manager.close()
    single_manager.close()
    print("✓ Cleanup completed")
    
    # Final verdict
    print("\n" + "="*50)
    print("FINAL VERDICT")
    print("="*50)
    
    correctness_passed = (not reset_differences and not final_differences)
    performance_improved = total_parallel_time < total_single_time
    
    if correctness_passed and performance_improved:
        print("🎉 ALL TESTS PASSED!")
        print("✓ Parallel execution produces identical results")
        print("✓ Parallel execution is faster than single-process")
    elif correctness_passed:
        print("⚠️  CORRECTNESS PASSED, PERFORMANCE ISSUE")
        print("✓ Parallel execution produces identical results")
        print("❌ Parallel execution is slower than single-process")
    elif performance_improved:
        print("⚠️  PERFORMANCE PASSED, CORRECTNESS ISSUE")
        print("❌ Parallel execution produces different results")
        print("✓ Parallel execution is faster than single-process")
    else:
        print("❌ TESTS FAILED")
        print("❌ Parallel execution produces different results")
        print("❌ Parallel execution is slower than single-process")
    
    return correctness_passed, performance_improved

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
                    'n_groups': [2],
                }
            }
        },
        'custom_envs': {
            'SimpleSokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 10,
                'env_config': {
                    'dim_room': [7, 7],
                    'num_boxes': 1,
                    'max_steps': 200
                }
            }
        },
        'seed': 42,
        'use_parallel_envs': True
    })

    # Test parallel execution
    es_manager = EnvStateManager(config, mode='train')
    print(f"Created EnvStateManager with {len(es_manager.envs)} environments")
    print(f"Using parallel execution: {es_manager.use_parallel}")

    # Reset and test
    rollout_cache = es_manager.reset(seed=123)
    print(f"Reset completed. Rollout cache length: {len(rollout_cache)}")

    # Test step
    all_env_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Go down",
            "llm_response": "Go down",
            "actions": ["down"]
        },
        {
            "env_id": 1,
            "llm_raw_response": "Go right",
            "llm_response": "Go right", 
            "actions": ["right"]
        }
    ]
    
    env_outputs = es_manager.step(all_env_inputs)
    print(f"Step completed. Active environments: {len(env_outputs)}")
    
    # Cleanup
    es_manager.close()
    print("Test completed!")

if __name__ == "__main__":
    # Run both tests
    print("Running basic parallel execution test...")
    test_parallel_execution()
    
    print("\n\n")
    
    print("Running comprehensive parallel vs single-process comparison...")
    test_parallel_vs_single_process() 