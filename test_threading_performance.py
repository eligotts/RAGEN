"""
Test script to benchmark threading performance in EnvStateManager
"""
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__)))

from omegaconf import OmegaConf
from ragen.llm_agent.es_manager import EnvStateManager

def create_test_config():
    """Create a simple test configuration"""
    return {
        'es_manager': {
            'enable_parallel_env_steps': True,
            'max_env_worker_threads': 4,
            'format_penalty': 0.1,
            'train': {
                'env_groups': 1,
                'group_size': 4,
                'env_configs': {
                    'tags': ['Bandit'],
                    'n_groups': [1]
                }
            }
        },
        'custom_envs': {
            'Bandit': {
                'env_type': 'bandit',
                'max_actions_per_traj': 1,
                'env_config': {
                    'lo_arm_name': 'Turtle',
                    'hi_arm_name': 'Falcon'
                }
            }
        }
    }

def run_performance_test():
    """Run performance comparison between sequential and parallel processing"""
    config = OmegaConf.create(create_test_config())
    
    print("🧪 Threading Performance Test")
    print("=" * 50)
    print(f"Number of environments: 4")
    print(f"Artificial delay per step: 100ms")
    print()
    
    # Test data - simulate 4 environments taking actions
    test_inputs = [
        {
            "env_id": 0,
            "llm_raw_response": "Falcon",
            "llm_response": "Falcon", 
            "actions": ["Falcon"]
        },
        {
            "env_id": 1,
            "llm_raw_response": "Falcon",
            "llm_response": "Falcon",
            "actions": ["Falcon"]
        },
        {
            "env_id": 2,
            "llm_raw_response": "Turtle", 
            "llm_response": "Turtle",
            "actions": ["Turtle"]
        },
        {
            "env_id": 3,
            "llm_raw_response": "Falcon",
            "llm_response": "Falcon", 
            "actions": ["Falcon"]
        }
    ]
    
    # Test sequential processing
    print("🔄 Testing Sequential Processing...")
    es_manager_seq = EnvStateManager(config, mode="train")
    es_manager_seq.enable_parallel_env_steps = False
    es_manager_seq.reset(seed=123)
    
    start_time = time.time()
    env_outputs_seq = es_manager_seq.step(test_inputs)
    seq_time = time.time() - start_time
    
    print(f"   ⏱️  Sequential time: {seq_time:.3f}s")
    print(f"   📊 Active environments: {len(env_outputs_seq)}")
    es_manager_seq.close()
    
    # Test parallel processing  
    print("\n⚡ Testing Parallel Processing...")
    es_manager_par = EnvStateManager(config, mode="train")
    es_manager_par.enable_parallel_env_steps = True
    es_manager_par.reset(seed=123)
    
    start_time = time.time()
    env_outputs_par = es_manager_par.step(test_inputs)
    par_time = time.time() - start_time
    
    print(f"   ⏱️  Parallel time: {par_time:.3f}s") 
    print(f"   📊 Active environments: {len(env_outputs_par)}")
    es_manager_par.close()
    
    # Results
    print("\n📈 Performance Results:")
    print("=" * 50)
    speedup = seq_time / par_time if par_time > 0 else float('inf')
    print(f"Sequential: {seq_time:.3f}s")
    print(f"Parallel:   {par_time:.3f}s") 
    print(f"Speedup:    {speedup:.2f}x")
    
    if speedup > 2.0:
        print("✅ Threading provides significant speedup!")
    elif speedup > 1.2:
        print("⚠️  Threading provides some speedup")
    else:
        print("❌ Threading may not be working as expected")
        
    return seq_time, par_time, speedup

if __name__ == "__main__":
    try:
        run_performance_test()
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc() 