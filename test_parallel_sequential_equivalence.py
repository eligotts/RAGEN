#!/usr/bin/env python3
"""
Comprehensive test script for parallel vs sequential equivalence in EnvStateManager.
Tests all environment types to ensure parallel processing produces identical results.
"""

import os
import sys
import json
import time
import traceback
from typing import Dict, List, Any, Optional
import numpy as np
from dataclasses import asdict

# Add the project root to the path
sys.path.append('.')

try:
    from ragen.llm_agent.es_manager import EnvStateManager
    import hydra
    from omegaconf import DictConfig, OmegaConf
except ImportError as e:
    print(f"Failed to import required modules: {e}")
    print("Make sure you're running this from the RAGEN project root directory")
    sys.exit(1)

class ParallelSequentialTester:
    """Test class for comparing parallel vs sequential processing results"""
    
    def __init__(self):
        self.test_results = {}
        self.environment_configs = self._get_environment_configs()
        
    def _get_environment_configs(self) -> Dict[str, Dict]:
        """Get test configurations for all environment types"""
        return {
            'bandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': {
                    'lo_arm_name': "Phoenix",
                    'hi_arm_name': "Dragon"
                }
            },
            'countdown': {
                'env_type': 'countdown',
                'max_actions_per_traj': 1,
                'env_config': None
            },
            'sokoban': {
                'env_type': 'sokoban',
                'max_actions_per_traj': 10,
                'env_config': {
                    'dim_x': 6,
                    'dim_y': 6,
                    'num_boxes': 1,
                    'max_steps': 100
                }
            },
            'frozen_lake': {
                'env_type': 'frozen_lake',
                'max_actions_per_traj': 10,
                'env_config': None
            },
            'metamathqa': {
                'env_type': 'metamathqa',
                'max_actions_per_traj': 1,
                'env_config': None
            }
        }
    
    def _create_test_config(self, env_name: str, env_config: Dict) -> DictConfig:
        """Create a test configuration for a specific environment"""
        config_dict = {
            'es_manager': {
                'train': {
                    'env_groups': 2,
                    'group_size': 2,
                    'env_configs': {
                        'tags': [env_name],
                        'n_groups': [1]
                    }
                },
                'val': {
                    'env_groups': 2,
                    'group_size': 2,
                    'env_configs': {
                        'tags': [env_name],
                        'n_groups': [1]
                    }
                },
                'format_penalty': -0.1,
                'use_multiprocessing': True,
                'max_workers': 2,
                'sequential_env_types': []  # Test all environments in parallel
            },
            'custom_envs': {
                env_name: {
                    'env_type': env_config['env_type'],
                    'max_actions_per_traj': env_config['max_actions_per_traj'],
                    'env_instruction': f"Test instruction for {env_name}",
                    'max_tokens': 100,
                    'env_config': env_config['env_config']
                }
            }
        }
        return OmegaConf.create(config_dict)
    
    def _generate_test_actions(self, env_name: str, num_steps: int = 3) -> List[List[Dict]]:
        """Generate deterministic test actions for each environment"""
        if env_name == 'bandit':
            # Bandit: simple arm selection
            return [
                [{'env_id': 0, 'llm_raw_response': 'Choose Dragon', 'llm_response': 'Choose Dragon', 'actions': ['2']}],
                [{'env_id': 0, 'llm_raw_response': 'Choose Phoenix', 'llm_response': 'Choose Phoenix', 'actions': ['1']}],
                [{'env_id': 0, 'llm_raw_response': 'Choose Dragon again', 'llm_response': 'Choose Dragon again', 'actions': ['2']}]
            ]
        
        elif env_name == 'countdown':
            # Countdown: mathematical expressions
            return [
                [{'env_id': 0, 'llm_raw_response': '2 + 3', 'llm_response': '2 + 3', 'actions': ['2 + 3']}],
                [{'env_id': 0, 'llm_raw_response': '5 - 1', 'llm_response': '5 - 1', 'actions': ['5 - 1']}],
                [{'env_id': 0, 'llm_raw_response': '4 * 2', 'llm_response': '4 * 2', 'actions': ['4 * 2']}]
            ]
        
        elif env_name == 'sokoban':
            # Sokoban: movement actions
            return [
                [{'env_id': 0, 'llm_raw_response': 'Move right', 'llm_response': 'Move right', 'actions': ['right']}],
                [{'env_id': 0, 'llm_raw_response': 'Move down', 'llm_response': 'Move down', 'actions': ['down']}],
                [{'env_id': 0, 'llm_raw_response': 'Move left', 'llm_response': 'Move left', 'actions': ['left']}]
            ]
        
        elif env_name == 'frozen_lake':
            # FrozenLake: movement actions
            return [
                [{'env_id': 0, 'llm_raw_response': 'Move right', 'llm_response': 'Move right', 'actions': ['3']}],
                [{'env_id': 0, 'llm_raw_response': 'Move down', 'llm_response': 'Move down', 'actions': ['2']}],
                [{'env_id': 0, 'llm_raw_response': 'Move left', 'llm_response': 'Move left', 'actions': ['1']}]
            ]
        
        elif env_name == 'metamathqa':
            # MetaMathQA: mathematical reasoning
            return [
                [{'env_id': 0, 'llm_raw_response': '2 + 2 = 4', 'llm_response': '2 + 2 = 4', 'actions': ['2 + 2 = 4']}],
                [{'env_id': 0, 'llm_raw_response': '3 * 4 = 12', 'llm_response': '3 * 4 = 12', 'actions': ['3 * 4 = 12']}],
                [{'env_id': 0, 'llm_raw_response': '10 / 2 = 5', 'llm_response': '10 / 2 = 5', 'actions': ['10 / 2 = 5']}]
            ]
        
        else:
            # Default: generic actions
            return [
                [{'env_id': 0, 'llm_raw_response': 'Action 1', 'llm_response': 'Action 1', 'actions': ['action1']}],
                [{'env_id': 0, 'llm_raw_response': 'Action 2', 'llm_response': 'Action 2', 'actions': ['action2']}],
                [{'env_id': 0, 'llm_raw_response': 'Action 3', 'llm_response': 'Action 3', 'actions': ['action3']}]
            ]
    
    def _run_single_test(self, env_name: str, env_config: Dict, test_seed: int = 12345) -> Dict[str, Any]:
        """Run a single test comparing parallel vs sequential for one environment"""
        print(f"\n{'='*60}")
        print(f"Testing environment: {env_name}")
        print(f"{'='*60}")
        
        test_result = {
            'environment': env_name,
            'success': False,
            'error': None,
            'sequential_results': None,
            'parallel_results': None,
            'results_match': False,
            'execution_time': {}
        }
        
        try:
            # Create test configuration
            config = self._create_test_config(env_name, env_config)
            
            # Generate test actions
            test_actions = self._generate_test_actions(env_name)
            
            # Test 1: Sequential Processing
            print(f"Running sequential test for {env_name}...")
            start_time = time.time()
            
            config.es_manager.use_multiprocessing = False
            sequential_manager = EnvStateManager(config, mode="train")
            sequential_results = self._run_environment_test(sequential_manager, test_seed, test_actions)
            sequential_manager.close()
            
            sequential_time = time.time() - start_time
            test_result['execution_time']['sequential'] = sequential_time
            test_result['sequential_results'] = sequential_results
            
            print(f"Sequential test completed in {sequential_time:.3f}s")
            
            # Test 2: Parallel Processing
            print(f"Running parallel test for {env_name}...")
            start_time = time.time()
            
            config.es_manager.use_multiprocessing = True
            parallel_manager = EnvStateManager(config, mode="train")
            parallel_results = self._run_environment_test(parallel_manager, test_seed, test_actions)
            parallel_manager.close()
            
            parallel_time = time.time() - start_time
            test_result['execution_time']['parallel'] = parallel_time
            test_result['parallel_results'] = parallel_results
            
            print(f"Parallel test completed in {parallel_time:.3f}s")
            
            # Compare results
            results_match = self._compare_results(sequential_results, parallel_results)
            test_result['results_match'] = results_match
            test_result['success'] = True
            
            if results_match:
                print(f"✅ {env_name}: Results MATCH between parallel and sequential")
                print(f"   Speedup: {parallel_time/sequential_time:.2f}x")
            else:
                print(f"❌ {env_name}: Results DO NOT MATCH between parallel and sequential")
                self._print_differences(sequential_results, parallel_results)
            
        except Exception as e:
            error_msg = f"Error testing {env_name}: {str(e)}"
            print(f"❌ {error_msg}")
            test_result['error'] = error_msg
            traceback.print_exc()
        
        return test_result
    
    def _run_environment_test(self, manager: EnvStateManager, seed: int, test_actions: List[List[Dict]]) -> Dict[str, Any]:
        """Run a complete test sequence for an environment manager"""
        results = {
            'initial_state': None,
            'step_results': [],
            'final_state': None,
            'rollout_states': None
        }
        
        # Reset environment
        rollout_cache = manager.reset(seed=seed)
        results['initial_state'] = self._extract_state_info(rollout_cache)
        
        # Run steps
        for i, actions in enumerate(test_actions):
            try:
                step_outputs = manager.step(actions)
                results['step_results'].append({
                    'step': i,
                    'actions': actions,
                    'outputs': self._extract_step_info(step_outputs)
                })
            except Exception as e:
                results['step_results'].append({
                    'step': i,
                    'actions': actions,
                    'error': str(e)
                })
        
        # Get final state
        try:
            final_rollout = manager.get_rollout_states()
            results['final_state'] = self._extract_state_info(final_rollout)
            results['rollout_states'] = final_rollout
        except Exception as e:
            results['final_state'] = {'error': str(e)}
        
        return results
    
    def _extract_state_info(self, rollout_cache: List[Dict]) -> Dict[str, Any]:
        """Extract key state information for comparison"""
        if not rollout_cache:
            return {'empty': True}
        
        state_info = {
            'num_environments': len(rollout_cache),
            'environment_states': []
        }
        
        for i, cache in enumerate(rollout_cache):
            env_state = {
                'env_id': cache.get('env_id'),
                'group_id': cache.get('group_id'),
                'tag': cache.get('tag'),
                'penalty': cache.get('penalty', 0),
                'history_length': len(cache.get('history', [])),
                'metrics': cache.get('metrics', {}),
                'last_state': self._extract_last_state(cache.get('history', []))
            }
            state_info['environment_states'].append(env_state)
        
        return state_info
    
    def _extract_last_state(self, history: List[Dict]) -> Dict[str, Any]:
        """Extract information from the last state in history"""
        if not history:
            return {'empty': True}
        
        last_entry = history[-1]
        return {
            'state_type': type(last_entry.get('state')).__name__,
            'actions_left': last_entry.get('actions_left'),
            'has_actions': 'actions' in last_entry,
            'has_reward': 'reward' in last_entry,
            'has_info': 'info' in last_entry
        }
    
    def _extract_step_info(self, step_outputs: List[Dict]) -> List[Dict]:
        """Extract key information from step outputs"""
        info = []
        for output in step_outputs:
            output_info = {
                'env_id': output.get('env_id'),
                'history_length': len(output.get('history', [])),
                'penalty': output.get('penalty', 0)
            }
            info.append(output_info)
        return info
    
    def _compare_results(self, sequential_results: Dict, parallel_results: Dict) -> bool:
        """Compare sequential and parallel results for equality"""
        try:
            # Compare initial states
            if not self._compare_state_info(sequential_results['initial_state'], parallel_results['initial_state']):
                return False
            
            # Compare step results
            if len(sequential_results['step_results']) != len(parallel_results['step_results']):
                return False
            
            for seq_step, par_step in zip(sequential_results['step_results'], parallel_results['step_results']):
                if not self._compare_step_results(seq_step, par_step):
                    return False
            
            # Compare final states
            if not self._compare_state_info(sequential_results['final_state'], parallel_results['final_state']):
                return False
            
            return True
            
        except Exception as e:
            print(f"Error during comparison: {e}")
            return False
    
    def _compare_state_info(self, seq_state: Dict, par_state: Dict) -> bool:
        """Compare state information for equality"""
        if seq_state.get('empty') and par_state.get('empty'):
            return True
        
        if seq_state.get('num_environments') != par_state.get('num_environments'):
            return False
        
        seq_envs = seq_state.get('environment_states', [])
        par_envs = par_state.get('environment_states', [])
        
        if len(seq_envs) != len(par_envs):
            return False
        
        for seq_env, par_env in zip(seq_envs, par_envs):
            if not self._compare_environment_state(seq_env, par_env):
                return False
        
        return True
    
    def _compare_environment_state(self, seq_env: Dict, par_env: Dict) -> bool:
        """Compare individual environment states"""
        # Compare basic fields
        basic_fields = ['env_id', 'group_id', 'tag', 'penalty', 'history_length']
        for field in basic_fields:
            if seq_env.get(field) != par_env.get(field):
                return False
        
        # Compare metrics (allow small floating point differences)
        seq_metrics = seq_env.get('metrics', {})
        par_metrics = par_env.get('metrics', {})
        
        if set(seq_metrics.keys()) != set(par_metrics.keys()):
            return False
        
        for key in seq_metrics:
            seq_val = seq_metrics[key]
            par_val = par_metrics[key]
            
            if isinstance(seq_val, (int, float)) and isinstance(par_val, (int, float)):
                if abs(seq_val - par_val) > 1e-6:
                    return False
            elif seq_val != par_val:
                return False
        
        return True
    
    def _compare_step_results(self, seq_step: Dict, par_step: Dict) -> bool:
        """Compare step results for equality"""
        if seq_step.get('step') != par_step.get('step'):
            return False
        
        # Compare actions
        seq_actions = seq_step.get('actions', [])
        par_actions = par_step.get('actions', [])
        
        if len(seq_actions) != len(par_actions):
            return False
        
        for seq_action, par_action in zip(seq_actions, par_actions):
            if seq_action != par_action:
                return False
        
        # Compare outputs
        seq_outputs = seq_step.get('outputs', [])
        par_outputs = par_step.get('outputs', [])
        
        if len(seq_outputs) != len(par_outputs):
            return False
        
        for seq_output, par_output in zip(seq_outputs, par_outputs):
            if not self._compare_step_output(seq_output, par_output):
                return False
        
        return True
    
    def _compare_step_output(self, seq_output: Dict, par_output: Dict) -> bool:
        """Compare individual step outputs"""
        basic_fields = ['env_id', 'history_length', 'penalty']
        for field in basic_fields:
            if seq_output.get(field) != par_output.get(field):
                return False
        return True
    
    def _print_differences(self, sequential_results: Dict, parallel_results: Dict):
        """Print detailed differences between sequential and parallel results"""
        print("\nDetailed differences:")
        
        # Compare initial states
        if not self._compare_state_info(sequential_results['initial_state'], parallel_results['initial_state']):
            print("  ❌ Initial states differ")
        
        # Compare step results
        for i, (seq_step, par_step) in enumerate(zip(sequential_results['step_results'], parallel_results['step_results'])):
            if not self._compare_step_results(seq_step, par_step):
                print(f"  ❌ Step {i} results differ")
        
        # Compare final states
        if not self._compare_state_info(sequential_results['final_state'], parallel_results['final_state']):
            print("  ❌ Final states differ")
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run tests for all environment types"""
        print("Starting comprehensive parallel vs sequential equivalence tests...")
        print(f"Testing {len(self.environment_configs)} environment types")
        
        all_results = {
            'test_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'total_environments': len(self.environment_configs),
            'successful_tests': 0,
            'failed_tests': 0,
            'matching_results': 0,
            'environment_results': {}
        }
        
        for env_name, env_config in self.environment_configs.items():
            try:
                result = self._run_single_test(env_name, env_config)
                all_results['environment_results'][env_name] = result
                
                if result['success']:
                    all_results['successful_tests'] += 1
                    if result['results_match']:
                        all_results['matching_results'] += 1
                else:
                    all_results['failed_tests'] += 1
                    
            except Exception as e:
                print(f"❌ Failed to test {env_name}: {e}")
                all_results['failed_tests'] += 1
                all_results['environment_results'][env_name] = {
                    'environment': env_name,
                    'success': False,
                    'error': str(e)
                }
        
        # Print summary
        self._print_summary(all_results)
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _print_summary(self, results: Dict[str, Any]):
        """Print test summary"""
        print(f"\n{'='*80}")
        print("TEST SUMMARY")
        print(f"{'='*80}")
        print(f"Timestamp: {results['test_timestamp']}")
        print(f"Total environments tested: {results['total_environments']}")
        print(f"Successful tests: {results['successful_tests']}")
        print(f"Failed tests: {results['failed_tests']}")
        print(f"Results match: {results['matching_results']}/{results['successful_tests']}")
        
        print(f"\nDetailed Results:")
        for env_name, result in results['environment_results'].items():
            status = "✅ PASS" if result.get('success') and result.get('results_match') else "❌ FAIL"
            error = f" ({result.get('error', '')})" if result.get('error') else ""
            print(f"  {env_name}: {status}{error}")
            
            if result.get('success') and result.get('execution_time'):
                seq_time = result['execution_time'].get('sequential', 0)
                par_time = result['execution_time'].get('parallel', 0)
                if seq_time > 0:
                    speedup = par_time / seq_time
                    print(f"    Speedup: {speedup:.2f}x")
    
    def _save_results(self, results: Dict[str, Any]):
        """Save test results to file"""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        filename = f"parallel_sequential_test_results_{timestamp}.json"
        
        # Convert results to JSON-serializable format
        serializable_results = self._make_serializable(results)
        
        try:
            with open(filename, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            print(f"\nResults saved to: {filename}")
        except Exception as e:
            print(f"Failed to save results: {e}")
    
    def _make_serializable(self, obj):
        """Convert object to JSON-serializable format"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return self._make_serializable(obj.__dict__)
        else:
            return obj

def main():
    """Main test function"""
    print("RAGEN Parallel vs Sequential Equivalence Test")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('ragen'):
        print("Error: 'ragen' directory not found. Please run this script from the RAGEN project root.")
        sys.exit(1)
    
    # Run tests
    tester = ParallelSequentialTester()
    results = tester.run_all_tests()
    
    # Exit with appropriate code
    if results['failed_tests'] > 0:
        print(f"\n❌ {results['failed_tests']} tests failed!")
        sys.exit(1)
    elif results['matching_results'] < results['successful_tests']:
        print(f"\n⚠️  {results['successful_tests'] - results['matching_results']} tests have different results!")
        sys.exit(1)
    else:
        print(f"\n✅ All tests passed! Parallel and sequential processing produce identical results.")
        sys.exit(0)

if __name__ == "__main__":
    main() 