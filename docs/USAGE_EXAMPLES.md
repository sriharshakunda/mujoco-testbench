# Usage Examples

This document provides practical examples for common tasks using the MuJoCo IL/RL pipeline.

## Table of Contents

1. [Basic Environment Usage](#basic-environment-usage)
2. [Data Collection](#data-collection)
3. [Imitation Learning Training](#imitation-learning-training)
4. [Policy Evaluation](#policy-evaluation)
5. [Custom Tasks](#custom-tasks)
6. [Advanced Teleoperation](#advanced-teleoperation)
7. [RL Fine-Tuning](#rl-fine-tuning)

## Basic Environment Usage

### Creating and Interacting with Environment

```python
from src.environment import PiperEnv, ReachingTask, TaskConfig
import numpy as np

# Create environment
env = PiperEnv(task_name="reaching", max_episode_steps=100, render_mode="human")

# Create and assign task
task_config = TaskConfig(
    name="reaching",
    max_episode_steps=100,
    target_tolerance=0.02,
    success_reward=1.0,
)
task = ReachingTask(task_config)
env.set_task(task)

# Reset environment
observation, info = env.reset()
print(f"Initial observation shape: {observation.shape}")
print(f"Observation: {observation}")

# Step through environment
total_reward = 0.0
for step in range(100):
    # Generate random action (or policy action)
    action = np.random.randn(6) * 0.1  # Small random actions
    
    # Step environment
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if step % 20 == 0:
        print(f"Step {step}: reward={reward:.3f}, total={total_reward:.3f}")
    
    if terminated or truncated:
        print(f"Episode finished at step {step + 1}")
        break

env.close()
```

### Rendering Support

```python
# Render during teleoperation (human mode)
env = PiperEnv(render_mode="human")
env.reset()
for _ in range(100):
    action = np.random.randn(6) * 0.05
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()

# Headless rendering (for servers without display)
env = PiperEnv(render_mode=None)  # No visual output
```

## Data Collection

### Collecting Demonstration Data

```python
from src.environment import PiperEnv, ReachingTask, TaskConfig
from src.teleoperation.devices import DeviceManager
from src.teleoperation.mapping import JointVelocityMapper
import numpy as np

# Setup environment
env = PiperEnv(task_name="reaching", render_mode="human")
task = ReachingTask(TaskConfig(max_episode_steps=100))
env.set_task(task)

# Setup input device (mock for testing)
device_manager = DeviceManager()
device = device_manager.add_device("mock", name="mock_device", num_axes=6)

# Setup input mapping
mapper = JointVelocityMapper(num_joints=6)
mapper.set_deadzone(0.05)
mapper.set_gain(1.0)

# Collect trajectories
trajectories = []

for episode in range(5):
    print(f"Recording episode {episode + 1}/5")
    obs, _ = env.reset()
    
    episode_obs, episode_acts = [], []
    
    for step in range(100):
        # Get device input (in real scenario, from physical device)
        state = device.get_state()
        if state is None:
            action = np.zeros(6)
        else:
            command = mapper.map_input_to_command(state.axes, state.buttons)
            action = np.array([command.get(f"joint_{i}", 0.0) for i in range(6)])
        
        # Execute action
        obs, reward, terminated, truncated, info = env.step(action)
        episode_obs.append(obs.copy())
        episode_acts.append(action.copy())
        
        if terminated or truncated:
            break
    
    trajectories.append({
        "observations": np.array(episode_obs),
        "actions": np.array(episode_acts),
    })
    print(f"  Collected {len(episode_obs)} steps")

# Cleanup
env.close()
device_manager.close_all()

print(f"Total trajectories collected: {len(trajectories)}")
```

### Recording to Disk

```python
from src.data import LeRobotHDF5Dataset, TrajectoryMetadata
import h5py

# Initialize LeRobot dataset
dataset = LeRobotHDF5Dataset("data/demonstrations.hdf5", mode="w")

# Add trajectories
for episode_id, traj in enumerate(trajectories):
    metadata = TrajectoryMetadata(
        task_id="reaching_v1",
        episode_id=episode_id,
        timestamp=datetime.now().isoformat(),
    )
    dataset.add_trajectory(
        observations=traj["observations"],
        actions=traj["actions"],
        metadata=metadata,
    )

dataset.close()
```

## Imitation Learning Training

### Basic IL Training

```python
from src.training.il import create_policy_network, BehavioralCloningTrainer
from src.data import TrajectoryDataset, create_dataloader
import torch

# Load your collected trajectories (see Data Collection section)
trajectories = [...]  # List of trajectory dicts

# Create dataset and dataloader
dataset = TrajectoryDataset(trajectories)
train_loader = create_dataloader(
    trajectories,
    batch_size=32,
    shuffle=True,
    train_split=0.9,
)

# Create policy network
config = {
    "arch": "mlp",
    "hidden_dims": [256, 256],
    "activation": "relu",
    "dropout_rate": 0.1,
}
policy = create_policy_network(input_dim=18, output_dim=6, config=config)

# Create trainer
training_config = {
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "optimizer": "adam",
    "lr_scheduler": "linear",
}
trainer = BehavioralCloningTrainer(
    policy,
    device="cuda" if torch.cuda.is_available() else "cpu",
    config=training_config,
)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    train_loss = 0.0
    num_batches = 0
    
    for batch_obs, batch_actions in train_loader:
        loss = trainer._compute_loss(batch_obs, batch_actions)
        train_loss += loss.item()
        num_batches += 1
    
    avg_loss = train_loss / num_batches
    print(f"Epoch {epoch + 1}: loss={avg_loss:.4f}")
    
    # Save checkpoint periodically
    if (epoch + 1) % 10 == 0:
        checkpoint = {
            "model_state_dict": policy.state_dict(),
            "epoch": epoch,
            "config": config,
        }
        torch.save(checkpoint, f"models/policy_epoch_{epoch + 1}.pt")
```

### Training with Validation

```python
from src.training.il import create_policy_network, BehavioralCloningTrainer
from src.data import create_train_val_split
import torch

# Split data into train/val
trajectories = [...]
train_trajs, val_trajs = create_train_val_split(
    trajectories,
    train_ratio=0.8,
    seed=42,
)

# Create dataloaders
train_loader = create_dataloader(train_trajs, batch_size=32, shuffle=True)
val_loader = create_dataloader(val_trajs, batch_size=32, shuffle=False)

# Create policy and trainer
policy = create_policy_network(input_dim=18, output_dim=6, config=config)
trainer = BehavioralCloningTrainer(policy, device="cuda")

# Training with validation
best_val_loss = float("inf")
patience = 5
patience_counter = 0

for epoch in range(50):
    # Training
    train_loss = 0.0
    for batch_obs, batch_actions in train_loader:
        loss = trainer._compute_loss(batch_obs, batch_actions)
        train_loss += loss.item()
    
    # Validation
    val_loss = 0.0
    with torch.no_grad():
        for batch_obs, batch_actions in val_loader:
            loss = trainer._compute_loss(batch_obs, batch_actions)
            val_loss += loss.item()
    
    print(f"Epoch {epoch + 1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")
    
    # Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(policy.state_dict(), "models/policy_best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break
```

## Policy Evaluation

### Evaluating Trained Policy

```python
from src.evaluation import PolicyRollout
from src.environment import PiperEnv, ReachingTask, TaskConfig
import numpy as np

# Create environment
env = PiperEnv(task_name="reaching", max_episode_steps=100)
task = ReachingTask(TaskConfig(max_episode_steps=100))
env.set_task(task)

# Load trained policy
rollout = PolicyRollout(
    checkpoint_path="models/policy_best.pt",
    policy_type="il",
    device="cuda",
)

# Run evaluation
num_episodes = 10
results = rollout.rollout_multiple(
    env,
    num_episodes=num_episodes,
    max_steps=100,
    render=False,
)

# Compute metrics
episode_returns = [r.episode_return for r in results]
success_rate = np.mean([r.success for r in results])
avg_length = np.mean([r.episode_length for r in results])

print(f"Evaluation Results (n={num_episodes}):")
print(f"  Success Rate: {success_rate:.1%}")
print(f"  Mean Return: {np.mean(episode_returns):.3f} ± {np.std(episode_returns):.3f}")
print(f"  Mean Episode Length: {avg_length:.1f}")

env.close()
```

### Performance Report

```python
from src.evaluation import PolicyRollout
import json

# Run evaluation
results = rollout.rollout_multiple(env, num_episodes=20)

# Generate report
report = {
    "num_episodes": len(results),
    "success_rate": float(np.mean([r.success for r in results])),
    "mean_return": float(np.mean([r.episode_return for r in results])),
    "std_return": float(np.std([r.episode_return for r in results])),
    "mean_length": float(np.mean([r.episode_length for r in results])),
    "task_info": results[0].task_info if results else {},
}

# Save report
with open("results/evaluation_report.json", "w") as f:
    json.dump(report, f, indent=2)

print(json.dumps(report, indent=2))
```

## Custom Tasks

### Implementing a Custom Task

```python
from src.environment.tasks import Task, TaskConfig
import numpy as np
from typing import Dict, Tuple

class PickPlaceTask(Task):
    """Pick up object and place at target location."""
    
    def __init__(self, config: TaskConfig):
        super().__init__(config)
        self.target_position = np.array([0.4, 0.0, 0.3])
        self.object_position = np.array([0.3, 0.0, 0.01])
        self.max_distance = 1.0
    
    def reset(self, observation: np.ndarray) -> None:
        """Reset task state."""
        # Randomize target position
        self.target_position = np.array([
            np.random.uniform(0.2, 0.6),
            np.random.uniform(-0.2, 0.2),
            np.random.uniform(0.2, 0.5),
        ])
    
    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        done: bool,
    ) -> Tuple[float, Dict]:
        """Compute pick-place reward."""
        ee_position = observation[-3:]
        
        # Distance to object (for picking)
        distance_to_object = np.linalg.norm(ee_position - self.object_position)
        pick_reward = -distance_to_object * 0.5
        
        # Distance to target (for placing)
        distance_to_target = np.linalg.norm(self.object_position - self.target_position)
        place_reward = -distance_to_target * 0.3
        
        # Success bonus
        success_bonus = 0.0
        if distance_to_target < self.config.target_tolerance:
            success_bonus = self.config.success_reward
        
        reward = pick_reward + place_reward + success_bonus - self.config.step_penalty
        
        return reward * self.config.reward_scale, {
            "distance_to_object": float(distance_to_object),
            "distance_to_target": float(distance_to_target),
        }
    
    def is_success(self, observation: np.ndarray) -> bool:
        """Check if object is at target."""
        return np.linalg.norm(self.object_position - self.target_position) < self.config.target_tolerance
    
    def get_task_observation(self, observation: np.ndarray) -> Dict:
        """Get task-specific observation."""
        return {
            "object_position": self.object_position.copy(),
            "target_position": self.target_position.copy(),
            "distance_to_target": float(np.linalg.norm(
                self.object_position - self.target_position
            )),
        }

# Use custom task
custom_config = TaskConfig(name="pick_place", max_episode_steps=200)
custom_task = PickPlaceTask(custom_config)
env.set_task(custom_task)
```

## Advanced Teleoperation

### Using Real Input Device

```python
from src.teleoperation.devices import SpacemouseDevice
from src.teleoperation.mapping import EndEffectorVelocityMapper

# Initialize spacemouse device
device = SpacemouseDevice(name="spacemouse_pro")
if not device.initialize():
    print("Spacemouse not found!")
    exit(1)

# Use end-effector (Cartesian) control
mapper = EndEffectorVelocityMapper()
mapper.set_deadzone(0.05)
mapper.set_gain(0.5)

# Polling loop
while True:
    state = device.get_state()
    if state is None:
        break
    
    # Map to commands
    command = mapper.map_input_to_command(state.axes, state.buttons)
    print(f"vel_x={command.get('vel_x', 0):.3f}, "
          f"vel_y={command.get('vel_y', 0):.3f}")

device.close()
```

### Multi-Device Teleoperation

```python
from src.teleoperation.devices import DeviceManager

# Manage multiple devices
manager = DeviceManager()

# Try different device types
device1 = manager.add_device("spacemouse", name="spacemouse_primary")
device2 = manager.add_device("joystick", device_index=0, name="joystick_backup")

if device1 and device1.is_connected:
    # Use spacemouse if available
    state = manager.get_device_state("spacemouse_primary")
else:
    # Fall back to joystick
    state = manager.get_device_state("joystick_backup")

manager.close_all()
```

## RL Fine-Tuning

### IL Warmstart for RL

```python
from stable_baselines3 import PPO
from src.training.il import create_policy_network
import torch

# Load IL-trained policy weights
il_checkpoint = torch.load("models/policy_best.pt")
il_policy = create_policy_network(input_dim=18, output_dim=6, config=config)
il_policy.load_state_dict(il_checkpoint)

# Convert to stable-baselines3 format (requires policy adapter)
# This is a simplified example - real implementation needs proper conversion

# Train RL policy
rl_env = PiperEnv(task_name="reaching")

# Use IL-trained policy as initialization
model = PPO(
    "MlpPolicy",
    rl_env,
    learning_rate=1e-4,  # Lower learning rate for fine-tuning
    n_steps=1024,
    batch_size=64,
)

# Train for additional timesteps
model.learn(total_timesteps=100000)

# Save
model.save("models/policy_rl_finetuned")
```

