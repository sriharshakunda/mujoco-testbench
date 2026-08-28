# Configuration Reference

This document describes all configurable options for the MuJoCo IL/RL pipeline.

## Configuration Structure

Configurations are stored in YAML files in the `config/` directory. The general structure is:

```yaml
# Global settings
task: reaching
max_episode_steps: 100
logging:
  level: INFO

# Model architecture
model:
  arch: mlp
  hidden_dims: [256, 256]
  activation: relu
  dropout_rate: 0.1

# Training settings
training:
  epochs: 50
  batch_size: 32
  learning_rate: 1e-3
  optimizer: adam
  weight_decay: 1e-4
  lr_scheduler: linear

# Teleoperation settings
teleoperation:
  control_frame: end_effector
  num_joints: 6
  deadzone: 0.05
  gain: 1.0

# Evaluation settings
evaluation:
  num_episodes: 10
  max_episode_steps: 100
  render_interval: 5
```

## Parameter Reference

### Global Settings

#### `task` (str)
Name of the task to perform.
- **Options**: `"reaching"`, `"pushing"`, `"grasping"`
- **Default**: `"reaching"`
- **Example**: `task: reaching`

#### `max_episode_steps` (int)
Maximum number of steps per episode before episode termination.
- **Range**: 1-10000
- **Default**: 100
- **Example**: `max_episode_steps: 200`

### Logging Settings

#### `logging.level` (str)
Logging verbosity level.
- **Options**: `"DEBUG"`, `"INFO"`, `"WARNING"`, `"ERROR"`, `"CRITICAL"`
- **Default**: `"INFO"`
- **Example**: `level: DEBUG`

#### `logging.log_dir` (str)
Directory for log files.
- **Default**: `"logs/"`
- **Example**: `log_dir: ./experiment_logs`

### Model Configuration

#### `model.arch` (str)
Policy network architecture type.
- **Options**: `"mlp"`
- **Default**: `"mlp"`
- **Example**: `arch: mlp`

#### `model.hidden_dims` (list of int)
Hidden layer dimensions for MLP.
- **Default**: `[256, 256]`
- **Example**: `hidden_dims: [128, 128, 64]`
- **Notes**: Deeper networks (more layers) may need more data; too large networks may overfit

#### `model.activation` (str)
Activation function for hidden layers.
- **Options**: `"relu"`, `"tanh"`, `"elu"`, `"sigmoid"`
- **Default**: `"relu"`
- **Examples**:
  - `activation: relu` - Best for most tasks
  - `activation: tanh` - Alternative, bounded output
  - `activation: elu` - Variant of ReLU

#### `model.use_batch_norm` (bool)
Whether to use batch normalization.
- **Default**: `false`
- **Examples**:
  - `use_batch_norm: true` - Helps with training stability
  - `use_batch_norm: false` - Simpler, slightly faster

#### `model.use_layer_norm` (bool)
Whether to use layer normalization.
- **Default**: `false`
- **Notes**: Cannot combine with `use_batch_norm`

#### `model.dropout_rate` (float)
Dropout probability for regularization.
- **Range**: 0.0-1.0
- **Default**: 0.0
- **Examples**:
  - `dropout_rate: 0.0` - No dropout
  - `dropout_rate: 0.1` - Light regularization
  - `dropout_rate: 0.5` - Strong regularization

### Training Configuration

#### `training.epochs` (int)
Number of training epochs.
- **Range**: 1-1000
- **Default**: 50
- **Example**: `epochs: 100`

#### `training.batch_size` (int)
Batch size for training.
- **Range**: 1-1000
- **Default**: 32
- **Examples**:
  - `batch_size: 16` - Smaller batches, more stable but slower
  - `batch_size: 128` - Larger batches, faster but noisier gradients

#### `training.learning_rate` (float)
Learning rate for optimizer.
- **Range**: 1e-6 to 1.0
- **Default**: 1e-3
- **Examples**:
  - `learning_rate: 1e-2` - Faster learning, risk of instability
  - `learning_rate: 1e-3` - Standard choice
  - `learning_rate: 1e-4` - Slower learning, more stable

#### `training.optimizer` (str)
Optimizer algorithm.
- **Options**: `"adam"`, `"sgd"`, `"adamw"`
- **Default**: `"adam"`
- **Examples**:
  - `optimizer: adam` - Recommended, adaptive learning rate
  - `optimizer: sgd` - Classic, may need learning rate scheduling

#### `training.weight_decay` (float)
L2 regularization coefficient.
- **Range**: 0.0-1.0
- **Default**: 0.0
- **Examples**:
  - `weight_decay: 0.0` - No regularization
  - `weight_decay: 1e-4` - Light regularization

#### `training.lr_scheduler` (str)
Learning rate scheduling strategy.
- **Options**: `null`, `"linear"`, `"exponential"`, `"cosine"`
- **Default**: `null`
- **Examples**:
  - `lr_scheduler: null` - Constant learning rate
  - `lr_scheduler: linear` - Linearly decrease over epochs

#### `training.log_interval` (int)
Steps between logging updates.
- **Default**: 10
- **Example**: `log_interval: 20`

#### `training.checkpoint_interval` (int)
Epochs between checkpoint saves.
- **Default**: 10
- **Example**: `checkpoint_interval: 5`

#### `training.checkpoint_dir` (str)
Directory for saving checkpoints.
- **Default**: `"checkpoints/"`
- **Example**: `checkpoint_dir: ./models`

### Teleoperation Configuration

#### `teleoperation.control_frame` (str)
Control frame for teleoperation.
- **Options**: `"joint"`, `"end_effector"`, `"base"`
- **Default**: `"end_effector"`
- **Descriptions**:
  - `joint`: Direct joint velocity control
  - `end_effector`: Task-space (Cartesian) control
  - `base`: Base frame control

#### `teleoperation.num_joints` (int)
Number of robot joints.
- **Default**: 6 (Piper arm)
- **Example**: `num_joints: 6`

#### `teleoperation.deadzone` (float)
Input deadzone threshold (values below ignored).
- **Range**: 0.0-1.0
- **Default**: 0.05
- **Examples**:
  - `deadzone: 0.0` - No deadzone
  - `deadzone: 0.05` - Filter noise below 5%
  - `deadzone: 0.15` - Larger deadzone for noisy devices

#### `teleoperation.gain` (float)
Sensitivity scaling factor.
- **Range**: 0.0-10.0
- **Default**: 1.0
- **Examples**:
  - `gain: 0.5` - Half sensitivity (safer)
  - `gain: 1.0` - Standard sensitivity
  - `gain: 2.0` - Double sensitivity (faster)

#### `teleoperation.axis_mapping` (dict)
Custom mapping from device axes to robot commands.
- **Default**: First 6 axes map to joint velocities
- **Example**:
```yaml
axis_mapping:
  axis_0: joint_0
  axis_1: joint_1
  axis_2: joint_2
```

#### `teleoperation.scale_limits` (dict)
Velocity limits for commands.
- **Example**:
```yaml
scale_limits:
  joint_0: [-1.5, 1.5]
  joint_1: [-1.0, 1.0]
```

### Evaluation Configuration

#### `evaluation.num_episodes` (int)
Number of episodes for evaluation.
- **Default**: 10
- **Example**: `num_episodes: 20`

#### `evaluation.max_episode_steps` (int)
Maximum steps per evaluation episode.
- **Default**: Same as global `max_episode_steps`
- **Example**: `max_episode_steps: 150`

#### `evaluation.render_interval` (int)
Render every N episodes (0 = never, 1 = always).
- **Default**: 5
- **Examples**:
  - `render_interval: 0` - No rendering
  - `render_interval: 1` - Render all episodes
  - `render_interval: 10` - Render every 10th episode

## Example Configurations

### Fast Training (for testing)
```yaml
task: reaching
max_episode_steps: 50

model:
  hidden_dims: [64, 64]
  dropout_rate: 0.0

training:
  epochs: 10
  batch_size: 16
  learning_rate: 1e-2
  optimizer: adam
```

### Production Training (for high performance)
```yaml
task: reaching
max_episode_steps: 200

model:
  hidden_dims: [256, 256, 128]
  activation: relu
  dropout_rate: 0.1
  use_batch_norm: true

training:
  epochs: 100
  batch_size: 64
  learning_rate: 1e-3
  optimizer: adamw
  weight_decay: 1e-4
  lr_scheduler: cosine
  checkpoint_interval: 5
```

### Conservative Teleoperation (safe)
```yaml
task: reaching
teleoperation:
  control_frame: joint
  deadzone: 0.15
  gain: 0.5
  scale_limits:
    joint_0: [-1.0, 1.0]
    joint_1: [-1.0, 1.0]
    joint_2: [-1.0, 1.0]
```

### Aggressive Teleoperation (responsive)
```yaml
task: reaching
teleoperation:
  control_frame: end_effector
  deadzone: 0.02
  gain: 2.0
```

## Configuration Best Practices

1. **Start with defaults**: The default configuration works well for most tasks
2. **Tune learning rate**: Most critical hyperparameter, try 1e-2, 1e-3, 1e-4
3. **Increase batch size**: Larger batches are more stable, use what fits in memory
4. **Use batch norm**: Helps with training stability for deeper networks
5. **Add dropout**: If overfitting is observed (val loss > train loss)
6. **Learning rate schedule**: Use cosine annealing for long training runs
7. **Log verbosity**: Set to DEBUG only when troubleshooting

## Configuration Validation

The system validates configurations on load:

```python
from src.utils import load_config, validate_config

config = load_config("config/reaching.yaml")
errors = validate_config(config)

if errors:
    print("Configuration errors:")
    for error in errors:
        print(f"  - {error}")
```

## Environment Variables

Override config values via environment variables (optional):

```bash
export PIPER_TASK=pushing
export PIPER_LEARNING_RATE=1e-4
export PIPER_BATCH_SIZE=64

python scripts/main.py --config config/reaching.yaml
```

