# Troubleshooting Guide

This guide covers common issues and their solutions.

## Installation Issues

### Issue: `ImportError: No module named 'mujoco'`

**Symptom**: Error when importing mujoco or running environment

**Solutions**:
```bash
# Install MuJoCo
pip install mujoco

# Verify installation
python -c "import mujoco; print(mujoco.__version__)"
```

**Alternative**: If pip install fails, download from [MuJoCo downloads](https://mujoco.org/download) and follow manual installation.

### Issue: `ImportError: No module named 'torch'`

**Symptom**: Error when importing torch or training modules

**Solutions**:
```bash
# Install PyTorch (CPU)
pip install torch torchvision torchaudio

# Install PyTorch with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
python -c "import torch; print(torch.__version__)"
```

### Issue: Dependency version conflicts

**Symptom**: Version mismatch errors during installation

**Solutions**:
```bash
# Recreate environment
python -m venv new_venv
source new_venv/bin/activate
pip install -r requirements.txt

# Or upgrade pip/setuptools
pip install --upgrade pip setuptools
```

## Device Input Issues

### Issue: `No devices found` or teleoperation not working

**Symptom**: Mock device works but real device fails

**Solutions**:

1. **Check device connection**:
```bash
# Linux: List USB devices
lsusb | grep -i 3d
lsusb | grep -i joystick

# macOS
system_profiler SPUSBDataType | grep -A 20 "3D"
```

2. **Try mock device first**:
```python
device_manager = DeviceManager()
device = device_manager.add_device("mock", name="test")
if device:
    print("Mock device works")
```

3. **Install device drivers**:
- **Spacemouse**: Install [pyspacemouse](https://github.com/pyspacemouse/pyspacemouse)
  ```bash
  pip install pyspacemouse
  ```
- **Joystick**: pygame handles most devices
  ```bash
  pip install pygame
  ```

4. **Check permissions (Linux)**:
```bash
# Add user to input group
sudo usermod -a -G input $USER

# Log out and back in for group changes to take effect
```

### Issue: Device detects but no input

**Symptom**: Device is found but `get_state()` returns zeros

**Solutions**:

1. **Increase deadzone**:
```python
mapper.set_deadzone(0.2)  # Ignore small noise
```

2. **Check device sensitivity**:
```python
device = DeviceManager().add_device("joystick")
state = device.get_state()
print(state.axes)  # Check raw axis values
```

3. **Recalibrate device**: Use device manufacturer's calibration software

## Environment and Task Issues

### Issue: `FileNotFoundError: URDF file not found`

**Symptom**: Error loading robot model

**Solutions**:
```python
# Use built-in Piper model (placeholder)
env = PiperEnv(task_name="reaching")

# If using custom URDF
from pathlib import Path
urdf_path = Path("data/piper.urdf")
if not urdf_path.exists():
    print(f"URDF not found at {urdf_path}")
    # Download from Agilex or provide your own
```

### Issue: `ValueError: Unknown task`

**Symptom**: Error when creating task

**Solutions**:
```python
# Use built-in tasks
from src.environment.tasks import create_task

task = create_task("reaching")  # Valid: reaching, pushing
# Invalid: create_task("custom_task")  # This will fail

# For custom tasks, implement Task subclass
from src.environment.tasks import Task
class CustomTask(Task):
    # ... implementation ...
```

### Issue: Episode doesn't terminate

**Symptom**: `rollout()` runs for entire `max_steps` without early termination

**Solutions**:
```python
# Check task success condition
task_obs = task.get_task_observation(observation)
is_success = task.is_success(observation)
print(f"Success: {is_success}, distance: {task_obs['distance_to_target']}")

# Adjust success threshold
task_config.target_tolerance = 0.01  # Make task easier to succeed
```

## Training Issues

### Issue: Loss doesn't decrease

**Symptom**: Training loss stays constant or increases

**Solutions**:

1. **Lower learning rate**:
```yaml
training:
  learning_rate: 1e-4  # Try smaller
```

2. **Check data quality**:
```python
# Visualize sample batch
loader = create_dataloader(trajectories, batch_size=4)
batch_obs, batch_actions = next(iter(loader))
print(f"Observation stats: mean={batch_obs.mean()}, std={batch_obs.std()}")
print(f"Action stats: mean={batch_actions.mean()}, std={batch_actions.std()}")
```

3. **Verify network architecture**:
```python
# Test forward pass
policy = create_policy_network(input_dim=18, output_dim=6, config=config)
test_obs = torch.randn(1, 18)
test_action = policy(test_obs)
print(f"Forward pass OK: {test_action.shape}")
```

4. **Check batch normalization**:
```yaml
model:
  use_batch_norm: false  # Try disabling
```

### Issue: Out of memory during training

**Symptom**: `RuntimeError: CUDA out of memory`

**Solutions**:

1. **Reduce batch size**:
```yaml
training:
  batch_size: 16  # Smaller batches
```

2. **Reduce model size**:
```yaml
model:
  hidden_dims: [128, 128]  # Fewer/smaller layers
```

3. **Use CPU training**:
```python
trainer = BehavioralCloningTrainer(policy, device="cpu")
```

4. **Enable gradient checkpointing** (advanced):
```python
# Clear GPU cache
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
```

### Issue: Training too slow

**Symptom**: Each epoch takes very long

**Solutions**:

1. **Use GPU**:
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
trainer = BehavioralCloningTrainer(policy, device=device)
```

2. **Increase batch size**:
```yaml
training:
  batch_size: 128  # Larger batches train faster
```

3. **Reduce dataset size** (testing only):
```python
# Use subset for debugging
small_trajs = trajectories[:10]  # First 10 trajectories
```

4. **Disable validation**:
```python
# Skip validation during training if just testing
```

### Issue: Model overfits (val_loss >> train_loss)

**Symptom**: Training loss is low but validation loss is high

**Solutions**:

1. **Add dropout**:
```yaml
model:
  dropout_rate: 0.3  # Increase regularization
```

2. **Add L2 regularization**:
```yaml
training:
  weight_decay: 1e-3  # Increase L2
```

3. **Early stopping**:
```python
# Stop when validation loss doesn't improve
if val_loss > best_val_loss:
    patience_counter += 1
    if patience_counter >= 5:
        break
```

4. **Collect more data**:
```python
# More diverse data helps generalization
trajectories = collect_more_data(num_episodes=50)
```

## Evaluation Issues

### Issue: Policy performance is poor

**Symptom**: Success rate < 50% or random-like behavior

**Solutions**:

1. **Verify training completed**:
```python
import torch
checkpoint = torch.load("models/policy.pt")
if "state_dict" in checkpoint:
    print("Checkpoint found and loaded")
else:
    print("Invalid checkpoint format")
```

2. **Check policy on simple data**:
```python
# Test on synthetic data
policy.eval()
test_obs = torch.randn(1, 18)
action = policy(test_obs)
print(f"Action magnitude: {torch.norm(action)}")
```

3. **Evaluate on training data** (sanity check):
```python
# If it fails on training data too, model is broken
rollout.policy = policy
for traj in trajectories[:3]:
    obs = traj["observations"][0]
    action = rollout.get_action(obs)
```

4. **Collect more high-quality demonstrations**:
```python
# Poor data quality hurts IL performance
# Retry with better-controlled teleoperation
```

### Issue: Rendering doesn't show

**Symptom**: `render_mode="human"` but no window appears

**Solutions**:

1. **Check display configuration**:
```bash
# Linux
echo $DISPLAY  # Should be :0 or :1

# If empty, set manually
export DISPLAY=:0
```

2. **Try headless rendering**:
```python
env = PiperEnv(render_mode=None)  # No render, just compute
```

3. **Docker X11 forwarding**:
```bash
docker run --rm \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  mujoco-piper:latest \
  python scripts/main.py --render
```

4. **Check MuJoCo viewer**:
```python
import mujoco
print(f"MuJoCo version: {mujoco.__version__}")
# If error, MuJoCo viewer might not be installed
```

## Docker Issues

### Issue: Docker build fails

**Symptom**: `docker build` exits with error

**Solutions**:
```bash
# Build with verbose output
docker build -v .

# Clear build cache if stuck
docker builder prune
docker build --no-cache .

# Check Dockerfile syntax
docker run --rm -i hadolint/hadolint < Dockerfile
```

### Issue: GPU not available in Docker

**Symptom**: `torch.cuda.is_available()` returns False

**Solutions**:
```bash
# Install nvidia-docker
apt-get install nvidia-docker2

# Use nvidia-docker
nvidia-docker run --rm -it mujoco-piper:latest python -c "import torch; print(torch.cuda.is_available())"

# Or use docker with --gpus
docker run --rm --gpus all mujoco-piper:latest python -c "import torch; print(torch.cuda.is_available())"
```

### Issue: Volume mounts don't work

**Symptom**: Files not accessible in container

**Solutions**:
```bash
# Use absolute paths
docker run --rm -v /full/path/data:/app/data mujoco-piper:latest

# Check mount
docker run --rm -v /tmp/test:/app/test mujoco-piper:latest ls -la /app/test

# Fix permissions
docker run --rm -v /tmp/test:/app/test:ro mujoco-piper:latest  # Read-only
```

## Performance Issues

### Issue: Slow data loading

**Symptom**: DataLoader is bottleneck during training

**Solutions**:

1. **Increase num_workers**:
```python
loader = create_dataloader(
    trajectories,
    batch_size=32,
    num_workers=4,  # Parallel loading
)
```

2. **Cache to RAM** (if dataset small):
```python
# Load entire dataset to memory
all_data = [traj for traj in trajectories]
```

3. **Use SSD** instead of HDD for data

## Miscellaneous Issues

### Issue: Configuration file not found

**Symptom**: `FileNotFoundError: config.yaml not found`

**Solutions**:
```bash
# Check if file exists
ls -la config/reaching.yaml

# Create from template
cp config/default.yaml config/reaching.yaml

# Use absolute path
python scripts/main.py --config /full/path/to/config.yaml
```

### Issue: Logging not working

**Symptom**: No log output or file

**Solutions**:
```python
from src.utils import setup_logging

# Setup logging explicitly
setup_logging(level="DEBUG", log_dir="logs/")

import logging
logger = logging.getLogger(__name__)
logger.info("This should now print")
```

## Getting More Help

If your issue isn't covered:

1. **Check error message carefully**: The traceback often points to the root cause
2. **Search documentation**: Most issues are in README.md, ARCHITECTURE.md, or docs/
3. **Run tests**: `pytest tests/test_integration.py -v` helps isolate issues
4. **Enable debug logging**: `setup_logging(level="DEBUG")`
5. **Minimize reproduction case**: Test with `MockDevice` and small dataset
6. **Check GitHub issues**: Similar issues might be reported

