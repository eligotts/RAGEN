# Parallel Environment Processing for RAGEN

This document describes the implementation of parallel environment processing in the RAGEN codebase, enabling true multi-process environment execution for improved training efficiency.

## Overview

The parallel environment processing system separates environment simulation from model training, allowing multiple environment instances to run in separate CPU processes while the model training happens on GPU. This architecture provides:

- **True Parallelism**: Environment simulation runs in separate processes, avoiding Python's GIL limitations
- **Scalability**: Support for hundreds of parallel environments
- **Efficiency**: Better CPU/GPU utilization by separating concerns
- **Compatibility**: Seamless integration with existing RAGEN training pipeline

## Architecture Components

### 1. Multi-Process Environment Container (`parallel_env_container.py`)

The core component that manages multiple environment instances in separate subprocesses.

**Key Features:**
- Spawns worker processes for each environment instance
- Handles inter-process communication via pipes
- Supports environment grouping for advanced algorithms (GRPO, GiGPO)
- Robust error handling and cleanup

**Usage:**
```python
from ragen.env.parallel_env_container import MultiProcessEnvironmentContainer

container = MultiProcessEnvironmentContainer(
    env_type='sokoban',
    env_num=4,      # Number of environment groups
    group_n=3,      # Environments per group
    seed=42,
    env_kwargs={'dim_room': (6, 6), 'num_boxes': 1}
)

# Reset all environments
obs_list, info_list = container.reset()

# Execute actions in parallel
actions = [1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4]  # 12 actions for 12 envs
obs_list, rewards, dones, infos = container.step(actions)

# Cleanup
container.close()
```

### 2. Environment Manager (`environment_manager.py`)

Provides a standardized interface between the training loop and multi-process environments.

**Key Features:**
- Converts raw environment observations to rich text prompts
- Maintains action history for context
- Handles action projection from LLM text to environment actions
- Environment-specific formatting (Sokoban, WebShop, etc.)

**Environment-Specific Managers:**
- `SokobanEnvironmentManager`: For Sokoban puzzle environments
- `WebShopEnvironmentManager`: For online shopping environments
- `CountdownEnvironmentManager`: For math puzzle environments
- `MetaMathQAEnvironmentManager`: For math QA environments

### 3. Action Projection Functions (`projection_functions.py`)

Converts LLM text output to valid environment actions.

**Supported Environments:**
- **Sokoban**: Maps text directions to integer actions
- **WebShop**: Validates search/click/buy action formats
- **Countdown**: Validates mathematical expressions
- **MetaMathQA**: Extracts numerical answers
- **FrozenLake**: Maps directions to discrete actions
- **Bandit**: Extracts arm selection numbers

**Example:**
```python
from ragen.env.projection_functions import create_projection_function

# Create projection function for Sokoban
projection_f = create_projection_function('sokoban')

# Convert LLM text to actions
text_actions = ["<action>up</action>", "<action>down</action>"]
actions, valids = projection_f(text_actions)
# actions: [1, 2], valids: [True, True]
```

### 4. Trajectory Collector (`trajectory_collector.py`)

Coordinates LLM generation with environment execution in multi-turn rollouts.

**Key Features:**
- Manages the interaction loop between LLM and environments
- Supports both vanilla and dynamic sampling strategies
- Handles batch size coordination
- Collects trajectory data for training

**Sampling Strategies:**
- **Vanilla**: Standard rollout collection
- **Dynamic**: Continues sampling until target batch size is met (for DAPO/GiGPO)

### 5. Environment Factory (`environment_factory.py`)

Centralized environment creation and configuration.

**Key Features:**
- Creates training and validation environments
- Handles environment-specific configuration
- Validates configuration consistency
- Supports different environment types

## Integration with Existing Codebase

### 1. Agent Trainer Integration

The `RayAgentTrainer` class has been updated to support parallel environments:

```python
# In agent_trainer.py
def init_agent_proxy(self):
    self.agent_proxy = LLMAgentProxy(...)
    
    # Initialize parallel environments if enabled
    if getattr(self.config, 'use_parallel_envs', True):
        self._init_parallel_environments()
```

### 2. Agent Proxy Integration

The `LLMAgentProxy` class automatically detects and uses parallel environments:

```python
def rollout(self, dataproto: DataProto, val=False):
    # Check if we have parallel environments available
    if hasattr(self, 'parallel_envs') and hasattr(self, 'traj_collector'):
        # Use new parallel environment system
        envs = self.val_parallel_envs if val else self.train_parallel_envs
        return self.traj_collector.multi_turn_loop(...)
    else:
        # Fallback to original system
        # ... original code ...
```

### 3. Configuration

The system uses existing RAGEN configuration structure:

```yaml
# In your config file
es_manager:
  train:
    env_groups: 4        # Number of environment groups
    group_size: 3        # Environments per group
    env_configs:
      tags: ["SokobanEnv"]
      n_groups: [1]
  val:
    env_groups: 2
    group_size: 1
    env_configs:
      tags: ["SokobanEnv"] 
      n_groups: [1]

custom_envs:
  SokobanEnv:
    env_type: sokoban
    max_actions_per_traj: 50
    env_config:
      dim_room: [6, 6]
      num_boxes: 1
      max_steps: 100
      search_depth: 30

# Optional: disable parallel environments
use_parallel_envs: false  # Default: true
```

## Advanced Features

### 1. Environment Grouping

Supports grouping environments for algorithms like GRPO and GiGPO:

```python
# 4 groups × 3 environments = 12 total environments
# Groups: [0,1,2], [3,4,5], [6,7,8], [9,10,11]
container = MultiProcessEnvironmentContainer(
    env_type='sokoban',
    env_num=4,      # Number of groups
    group_n=3,      # Environments per group
    seed=42
)
```

### 2. Dynamic Sampling

For advanced algorithms that require filtering based on reward variance:

```yaml
algorithm:
  filter_groups:
    enable: true
    max_num_gen_batches: 10
```

### 3. Reward Normalization

Supports various reward normalization strategies:

```yaml
agent_proxy:
  reward_normalization:
    grouping: "state"     # "state", "inductive", "batch"
    method: "mean_std"    # "mean_std", "mean", "asym_clip", "identity"
```

## Performance Considerations

### 1. Process Management

- Uses `spawn` context for cross-platform compatibility
- Automatic cleanup on process termination
- Graceful error handling and recovery

### 2. Memory Optimization

- History buffers are limited in size
- Efficient inter-process communication
- Minimal data copying between processes

### 3. Scalability

- Tested with up to 100+ parallel environments
- Linear scaling with number of CPU cores
- Configurable batch sizes for optimal performance

## Testing

Run the test script to verify the implementation:

```bash
python test_parallel_envs.py
```

The test script validates:
- Multi-process container functionality
- Environment manager operations
- Action projection functions
- Trajectory collector preprocessing

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed
2. **Process Hanging**: Check for proper cleanup in error cases
3. **Memory Issues**: Reduce batch size or number of parallel environments
4. **Configuration Errors**: Validate environment configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Fallback Mode

If parallel environments fail to initialize, the system automatically falls back to the original environment system:

```
Failed to initialize parallel environments: <error>
Falling back to original environment system
```

## Migration Guide

### From Original System

1. **No Code Changes Required**: The system automatically detects and uses parallel environments
2. **Configuration**: Update your config to specify environment groups and sizes
3. **Testing**: Run with small batch sizes first to verify functionality

### Performance Tuning

1. **Batch Size**: Set `env_groups × group_size` to match your desired batch size
2. **CPU Cores**: Use one environment per CPU core for optimal performance
3. **Memory**: Monitor memory usage with large numbers of environments

## Future Enhancements

1. **Shared Memory**: Use shared memory for large observation data
2. **GPU Environments**: Support for GPU-accelerated environments
3. **Distributed Environments**: Support for environments across multiple machines
4. **Environment Caching**: Cache environment states for faster resets

## Contributing

When adding new environment types:

1. Add environment class to `ragen/env/`
2. Register in `REGISTERED_ENVS` and `REGISTERED_ENV_CONFIGS`
3. Create projection function in `projection_functions.py`
4. Add environment manager in `environment_manager.py`
5. Update factory function in `environment_factory.py`
6. Add tests to verify functionality

## License

This implementation follows the same license as the RAGEN project. 