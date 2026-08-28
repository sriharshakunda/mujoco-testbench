# System Architecture

## Overview

This document describes the architecture of the MuJoCo IL/RL pipeline, including system design, module interactions, data flow, and key design decisions.

## System Design Philosophy

The pipeline follows these design principles:

1. **Modularity**: Each component (environment, teleoperation, data, training, evaluation) is independent and can be tested/replaced separately
2. **Abstraction**: Common interfaces (e.g., `InputDevice`, `Task`) allow implementations to be swapped
3. **Configuration-Driven**: Hyperparameters and behavior configured via YAML rather than hardcoded
4. **Production-Ready**: Comprehensive error handling, logging, and testing

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Pipeline Orchestrator (main.py)          │
└──────────────┬──────────────────────────────────────────────────┘
               │
       ┌───────┴────────┬──────────────┬──────────────┐
       │                │              │              │
       v                v              v              v
    DATA COLLECTION   IL TRAINING   RL TRAINING   EVALUATION
       │                │              │              │
       v                v              v              v
  ┌─────────────┬─────────────┬─────────────┬──────────────┐
  │ Environment │ Data Module │ Training    │ Evaluation   │
  │             │             │ Modules     │ Module       │
  └─────────────┴─────────────┴─────────────┴──────────────┘
       │              │              │              │
       v              v              v              v
  ┌────────────────────────────────────────────────────┐
  │            Core Components (src/)                  │
  ├────────────────────────────────────────────────────┤
  │ • environment/       - MuJoCo simulator            │
  │ • teleoperation/     - Device input abstraction    │
  │ • data/              - Dataset loading/processing  │
  │ • training/          - IL and RL trainers          │
  │ • evaluation/        - Policy evaluation           │
  │ • utils/             - Config, logging, helpers    │
  └────────────────────────────────────────────────────┘
```

## Module Descriptions

### 1. Environment Module (`src/environment/`)

**Purpose**: Provides a physics simulation environment for the Piper arm.

**Key Classes**:
- `PiperEnv`: Main environment wrapper with Gym-like interface
  - `reset()`: Reset to initial state
  - `step(action)`: Execute action, return state/reward/done
  - `render()`: Optional visualization
  - `close()`: Clean up resources

- `Task`: Abstract base class for manipulation tasks
  - `ReachingTask`: Move end-effector to target position
  - `PushingTask`: Push object to target location

- `TaskConfig`: Dataclass for task configuration
  - `name`: Task identifier
  - `max_episode_steps`: Episode length limit
  - `target_tolerance`: Success threshold
  - `reward_scale`: Reward scaling factor

**Data Flow**:
```
User Request
    ↓
PiperEnv.reset() → Observation
    ↓
PiperEnv.step(action) → (Observation, Reward, Done, Info)
    ↓
Task.compute_reward() → Reward value
```

### 2. Teleoperation Module (`src/teleoperation/`)

**Purpose**: Abstracts input device handling and command mapping.

**Key Classes**:
- `InputDevice` (Abstract): Base class for all input devices
  - `initialize()`: Initialize connection
  - `get_state()`: Read current device state
  - `close()`: Clean up

- Concrete Implementations:
  - `MockDevice`: Synthetic input for testing
  - `SpacemouseDevice`: 3D mouse input via pyspacemouse
  - `JoystickDevice`: Generic gamepad/joystick via pygame
  - `XboxControllerDevice`: Xbox-specific controller support

- `DeviceManager`: Manages multiple devices

- `InputToCommandMapper` (Abstract): Maps device input to robot commands
  - `JointVelocityMapper`: Joint-space control
  - `EndEffectorVelocityMapper`: Task-space (Cartesian) control
  - `ConfigurableMapper`: Load configuration from YAML

**Data Flow**:
```
Physical Device
    ↓
InputDevice.get_state() → DeviceState(axes, buttons, timestamp)
    ↓
InputToCommandMapper.map_input_to_command() → Command Dict
    ↓
Environment.step(command)
```

### 3. Data Module (`src/data/`)

**Purpose**: Handle trajectory collection, storage, and loading.

**Key Classes**:
- `TrajectoryDataset`: PyTorch Dataset for trajectory data
  - Converts list of trajectories to (observation, action) pairs
  - Supports indexing and batching

- `LeRobotHDF5Dataset`: HDF5-based storage with LeRobot format
  - Hierarchical storage: episodes → timesteps
  - Metadata: task_id, timestamp, episode_id
  - Efficient access for large datasets

- `TrajectoryRecorder`: Records trajectories during teleoperation
  - Buffering and persistence
  - Metadata tracking

**Data Flow**:
```
Collected Trajectories
    ↓
TrajectoryDataset (PyTorch Dataset)
    ↓
create_dataloader() → DataLoader with batching
    ↓
Training loop: for batch in loader:
```

### 4. Training Module (`src/training/`)

**Purpose**: Implement learning algorithms.

#### IL Submodule (`src/training/il/`)

- `MLPPolicy`: PyTorch neural network
  - Configurable hidden dimensions
  - Selectable activation functions
  - Optional batch normalization, layer norm, dropout

- `BehavioralCloningTrainer`: Supervised learning trainer
  - MSE loss for continuous action prediction
  - Validation on held-out data
  - Checkpoint saving/loading
  - Learning rate scheduling

**Training Flow**:
```
Policy Network (randomly initialized)
    ↓
For each epoch:
  For each batch from DataLoader:
    Forward pass: actions = policy(observations)
    Compute MSE loss: loss = ||actions - expert_actions||^2
    Backward pass and gradient update
    ↓
Checkpoint saved periodically
```

#### RL Submodule (`src/training/rl/`)
Placeholder for stable-baselines3 integration (PPO, SAC, etc.)

### 5. Evaluation Module (`src/evaluation/`)

**Purpose**: Run policies and compute performance metrics.

**Key Classes**:
- `PolicyRollout`: Execute policy in environment
  - `load_policy()`: Load from checkpoint
  - `get_action()`: Query policy for action
  - `rollout()`: Run single episode
  - `rollout_multiple()`: Run multiple episodes

- `RolloutResult`: Dataclass storing episode trajectory
  - `observations`: List of states
  - `actions`: List of executed actions
  - `rewards`: List of rewards
  - `episode_return`: Sum of rewards
  - `success`: Task success flag
  - `task_info`: Custom task metrics

**Evaluation Flow**:
```
Trained Policy Checkpoint
    ↓
PolicyRollout.load_policy()
    ↓
For each episode:
  env.reset()
  For each step:
    action = policy(observation)
    env.step(action)
    ↓
Collect RolloutResult
    ↓
Compute metrics: success_rate, mean_return, etc.
```

### 6. Utils Module (`src/utils/`)

**Purpose**: Shared utilities for configuration and logging.

- `config.py`: Configuration management
  - `load_config()`: Load YAML config files
  - `save_config()`: Save config to YAML
  - `merge_configs()`: Combine multiple configs
  - `validate_config()`: Schema validation

- `logging_utils.py`: Logging setup
  - `setup_logging()`: Configure logging
  - `get_logger()`: Get logger instance

## Data Structures

### Observation Format
```
[joint_positions (6), joint_velocities (6), ee_position (3), ee_velocity (3)]
Shape: (18,)
```

### Action Format
```
[joint_velocities (6)]
Shape: (6,)
```

### Trajectory Format
```python
{
    "observations": np.ndarray of shape (episode_length, 18),
    "actions": np.ndarray of shape (episode_length, 6),
}
```

### Config Structure
```yaml
task: reaching
max_episode_steps: 100

model:
  arch: mlp
  hidden_dims: [256, 256]
  activation: relu

training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3

teleoperation:
  control_frame: end_effector
  deadzone: 0.05
  gain: 1.0
```

## Key Design Decisions

### 1. Environment as Standalone Module
- **Decision**: Make `PiperEnv` independent of device input
- **Rationale**: Enables training without teleoperation, easy testing
- **Implication**: Data collection is separate pipeline stage

### 2. Configuration via YAML
- **Decision**: All hyperparameters in YAML config files
- **Rationale**: Reproducibility, easy experiment tracking
- **Implication**: Slight overhead in loading config, but much better for research

### 3. Behavioral Cloning in PyTorch (not Lightning)
- **Decision**: Custom training loop instead of PyTorch Lightning
- **Rationale**: BC is simple, Lightning adds unnecessary complexity
- **Implication**: More code, but better control and understanding

### 4. RL via Stable-Baselines3 (not custom)
- **Decision**: Use existing RL library rather than implementing from scratch
- **Rationale**: RL is complex, existing implementations are optimized and tested
- **Implication**: Must adapt interface, but greatly reduces bugs

### 5. Lazy-Loaded Data
- **Decision**: Use PyTorch DataLoader with lazy loading
- **Rationale**: Datasets may not fit in memory, standard practice
- **Implication**: Slightly slower access, but scales to large datasets

## Module Dependencies

```
environment/          (no dependencies)
    ↓
teleoperation/        (depends on: environment)
    ↓
data/                 (depends on: environment)
    ↓
training/             (depends on: data)
    ├── il/           (depends on: training, utils)
    └── rl/           (depends on: training, utils, environment)
    ↓
evaluation/           (depends on: environment, training/il, training/rl)
    ↓
utils/                (no core dependencies, used everywhere)
```

## Error Handling

Each module has consistent error handling:

1. **Input Validation**: Check parameters at entry points
2. **Graceful Degradation**: Fallback to default device if hardware unavailable
3. **Informative Logging**: Log errors with context
4. **Custom Exceptions**: Module-specific exceptions (future enhancement)

Example (teleoperation):
```python
device = device_manager.add_device("spacemouse")
if device is None:
    logger.warning("Spacemouse not found, falling back to mock device")
    device = device_manager.add_device("mock")
```

## Performance Considerations

### Memory
- Lazy loading: Only batch data loaded at time
- Efficient numpy arrays for trajectory storage
- PyTorch GPU memory management for training

### Computation
- Vectorized operations (numpy, PyTorch)
- Configurable batch sizes
- GPU acceleration support

### I/O
- HDF5 for efficient trajectory storage
- Checkpoint loading during evaluation
- Async data loading during training

## Testing Strategy

```
tests/test_integration.py:
├── TestModuleImports       - Verify all imports work
├── TestEnvironment         - Environment reset/step/render
├── TestTeleoperation       - Device input and mapping
├── TestTraining            - Policy creation and training
├── TestEvaluation          - Policy rollout and metrics
├── TestData                - Dataset creation and loading
├── TestConfiguration       - Config save/load
└── TestEndToEnd            - Full pipeline integration
```

## Future Extensions

1. **Vision-Based Control**: Add CNN policies for image observations
2. **Sim-to-Real Transfer**: Domain randomization utilities
3. **Multi-Task Learning**: Support training on multiple tasks simultaneously
4. **Online RL**: Integration with online learning algorithms
5. **Distributed Training**: Multi-GPU and multi-machine support
6. **Policy Distillation**: Compress trained policies for deployment

