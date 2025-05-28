#!/usr/bin/env python3
"""
Test script for parallel environment processing implementation.
This script tests the basic functionality of the parallel environment system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from transformers import AutoTokenizer
from verl import DataProto

# Test imports
try:
    from ragen.env.parallel_env_container import MultiProcessEnvironmentContainer
    from ragen.env.environment_manager import SokobanEnvironmentManager
    from ragen.env.projection_functions import create_projection_function
    from ragen.env.trajectory_collector import TrajectoryCollector
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)


def test_multiprocess_container():
    """Test the multi-process environment container."""
    print("\n=== Testing MultiProcessEnvironmentContainer ===")
    
    try:
        # Create container with Sokoban environments
        container = MultiProcessEnvironmentContainer(
            env_type='sokoban',
            env_num=2,
            group_n=2,
            seed=42,
            env_kwargs={'dim_room': (4, 4), 'num_boxes': 1, 'max_steps': 20}
        )
        
        # Test reset
        obs_list, info_list = container.reset()
        print(f"✓ Reset successful: {len(obs_list)} observations")
        
        # Test step
        actions = [1, 2, 3, 4]  # Up, Down, Left, Right
        obs_list, rewards, dones, infos = container.step(actions)
        print(f"✓ Step successful: {len(obs_list)} observations, rewards: {rewards}")
        
        # Test cleanup
        container.close()
        print("✓ Container cleanup successful")
        
    except Exception as e:
        print(f"✗ Container test failed: {e}")
        return False
    
    return True


def test_environment_manager():
    """Test the environment manager."""
    print("\n=== Testing EnvironmentManager ===")
    
    try:
        # Create container
        container = MultiProcessEnvironmentContainer(
            env_type='sokoban',
            env_num=1,
            group_n=2,
            seed=42,
            env_kwargs={'dim_room': (4, 4), 'num_boxes': 1, 'max_steps': 20}
        )
        
        # Create projection function
        projection_f = create_projection_function('sokoban')
        
        # Create environment manager
        env_manager = SokobanEnvironmentManager(container, projection_f, 'sokoban')
        
        # Test reset
        obs, infos = env_manager.reset()
        print(f"✓ Manager reset successful: {len(obs['text'])} text observations")
        
        # Test step with text actions
        text_actions = ["<action>up</action>", "<action>down</action>"]
        next_obs, rewards, dones, infos = env_manager.step(text_actions)
        print(f"✓ Manager step successful: rewards {rewards}, dones {dones}")
        
        # Test cleanup
        env_manager.close()
        print("✓ Manager cleanup successful")
        
    except Exception as e:
        print(f"✗ Manager test failed: {e}")
        return False
    
    return True


def test_projection_functions():
    """Test action projection functions."""
    print("\n=== Testing Projection Functions ===")
    
    try:
        # Test Sokoban projection
        sokoban_proj = create_projection_function('sokoban')
        text_actions = ["<action>up</action>", "<action>invalid</action>", "no action tags"]
        actions, valids = sokoban_proj(text_actions)
        print(f"✓ Sokoban projection: actions={actions}, valids={valids}")
        
        # Test WebShop projection
        webshop_proj = create_projection_function('webshop')
        text_actions = ["<action>search[shoes]</action>", "<action>click[buy]</action>"]
        actions, valids = webshop_proj(text_actions)
        print(f"✓ WebShop projection: actions={actions}, valids={valids}")
        
    except Exception as e:
        print(f"✗ Projection test failed: {e}")
        return False
    
    return True


def test_trajectory_collector():
    """Test the trajectory collector (basic functionality)."""
    print("\n=== Testing TrajectoryCollector ===")
    
    try:
        # Create mock config
        class MockConfig:
            def __init__(self):
                self.data = type('obj', (object,), {'max_prompt_length': 512})()
                self.env = type('obj', (object,), {'max_steps': 5})()
                self.env.rollout = type('obj', (object,), {'n': 2})()
        
        config = MockConfig()
        
        # Create tokenizer
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        # Create trajectory collector
        traj_collector = TrajectoryCollector(config, tokenizer)
        
        # Test preprocessing
        gen_batch = DataProto.from_single_dict({
            'input_ids': torch.tensor([[1, 2, 3], [4, 5, 6]]),
            'attention_mask': torch.tensor([[1, 1, 1], [1, 1, 1]])
        })
        
        obs = {
            'text': ['Test observation 1', 'Test observation 2'],
            'image': None,
            'anchor': ['anchor1', 'anchor2']
        }
        
        processed_batch = traj_collector.preprocess_batch(gen_batch, obs)
        print(f"✓ Trajectory collector preprocessing successful")
        
    except Exception as e:
        print(f"✗ Trajectory collector test failed: {e}")
        return False
    
    return True


def main():
    """Run all tests."""
    print("Starting parallel environment tests...")
    
    tests = [
        test_projection_functions,
        test_multiprocess_container,
        test_environment_manager,
        test_trajectory_collector,
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
    
    print(f"\n=== Test Results ===")
    print(f"Passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed! Parallel environment system is working.")
        return 0
    else:
        print("❌ Some tests failed. Check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 