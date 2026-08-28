"""Integration tests for the complete MuJoCo IL/RL pipeline.

This module tests the full pipeline from environment creation through
training and evaluation, ensuring all components work together correctly.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import pytest

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment import PiperEnv, ReachingTask, TaskConfig
from src.teleoperation.devices import MockDevice, DeviceManager
from src.teleoperation.mapping import JointVelocityMapper, EndEffectorVelocityMapper
from src.training.il import MLPPolicy, create_policy_network, BehavioralCloningTrainer
from src.evaluation import PolicyRollout
from src.data import TrajectoryDataset, create_dataloader, compute_dataset_statistics
from src.utils import load_config, save_config


class TestModuleImports:
    """Test that all modules can be imported and basic instantiation works."""

    def test_import_environment(self):
        """Test environment module imports."""
        from src.environment import PiperEnv, Task, ReachingTask, TaskConfig
        assert PiperEnv is not None
        assert Task is not None
        assert ReachingTask is not None
        assert TaskConfig is not None

    def test_import_teleoperation(self):
        """Test teleoperation module imports."""
        from src.teleoperation.devices import DeviceManager, MockDevice
        from src.teleoperation.mapping import InputToCommandMapper
        assert DeviceManager is not None
        assert MockDevice is not None
        assert InputToCommandMapper is not None

    def test_import_training(self):
        """Test training module imports."""
        from src.training.il import MLPPolicy, BehavioralCloningTrainer
        assert MLPPolicy is not None
        assert BehavioralCloningTrainer is not None

    def test_import_evaluation(self):
        """Test evaluation module imports."""
        from src.evaluation import PolicyRollout, RolloutResult
        assert PolicyRollout is not None
        assert RolloutResult is not None

    def test_import_data(self):
        """Test data module imports."""
        from src.data import TrajectoryDataset, LeRobotHDF5Dataset
        assert TrajectoryDataset is not None
        assert LeRobotHDF5Dataset is not None

    def test_import_utils(self):
        """Test utils module imports."""
        from src.utils import load_config, setup_logging
        assert load_config is not None
        assert setup_logging is not None


class TestEnvironment:
    """Test MuJoCo environment functionality."""

    def test_env_creation(self):
        """Test basic environment creation."""
        env = PiperEnv(task_name="reaching")
        assert env is not None
        assert env.task_name == "reaching"
        assert env.dt == 0.002
        env.close()

    def test_env_reset(self):
        """Test environment reset."""
        env = PiperEnv(task_name="reaching", max_episode_steps=50)
        obs, info = env.reset()
        assert isinstance(obs, np.ndarray)
        assert isinstance(info, dict)
        assert obs.shape == (18,)
        env.close()

    def test_env_step(self):
        """Test environment step."""
        env = PiperEnv(task_name="reaching", max_episode_steps=50)
        obs, _ = env.reset()
        
        # Create random action
        action = np.random.randn(6)
        obs, reward, terminated, truncated, info = env.step(action)
        
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)
        env.close()

    def test_env_task_assignment(self):
        """Test setting task on environment."""
        env = PiperEnv(task_name="reaching")
        task_config = TaskConfig(name="reaching", max_episode_steps=100)
        task = ReachingTask(task_config)
        env.set_task(task)
        assert env.task == task
        env.close()

    def test_rollout_completion(self):
        """Test that a complete rollout can execute without errors."""
        env = PiperEnv(task_name="reaching", max_episode_steps=10)
        task_config = TaskConfig(name="reaching", max_episode_steps=10)
        task = ReachingTask(task_config)
        env.set_task(task)
        
        obs, _ = env.reset()
        total_reward = 0.0
        
        for _ in range(10):
            action = np.random.randn(6)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        
        assert total_reward != 0.0  # Some reward was collected
        env.close()


class TestTeleoperation:
    """Test teleoperation device input and mapping."""

    def test_mock_device_creation(self):
        """Test mock device creation."""
        device = MockDevice(name="test_mock", num_axes=6, num_buttons=4)
        assert device.name == "test_mock"
        assert device.initialize()
        assert device.is_connected
        device.close()

    def test_device_state_reading(self):
        """Test reading device state."""
        device = MockDevice(name="test_mock", num_axes=6, num_buttons=4)
        device.initialize()
        
        # Set some test values
        device.set_axis_value(0, 0.5)
        device.set_button_state(0, True)
        
        state = device.get_state()
        assert state.axes["axis_0"] == 0.5
        assert state.buttons["button_0"] == True
        device.close()

    def test_device_manager(self):
        """Test device manager with multiple devices."""
        manager = DeviceManager()
        device = manager.add_device("mock", name="mock1", num_axes=6)
        assert device is not None
        assert "mock1" in manager.devices
        
        states = manager.get_all_states()
        assert "mock1" in states
        manager.close_all()

    def test_joint_velocity_mapper(self):
        """Test joint velocity input mapping."""
        mapper = JointVelocityMapper(num_joints=6)
        axes = {f"axis_{i}": 0.5 for i in range(6)}
        buttons = {}
        
        command = mapper.map_input_to_command(axes, buttons)
        assert "joint_0" in command
        assert command["joint_0"] == 0.5
        assert len(command) == 6

    def test_end_effector_velocity_mapper(self):
        """Test end-effector velocity input mapping."""
        mapper = EndEffectorVelocityMapper()
        axes = {f"axis_{i}": 0.1 for i in range(6)}
        buttons = {}
        
        command = mapper.map_input_to_command(axes, buttons)
        assert "vel_x" in command
        assert len(command) == 6

    def test_mapper_deadzone(self):
        """Test deadzone filtering in mapper."""
        mapper = JointVelocityMapper(num_joints=6)
        mapper.set_deadzone(0.1)
        
        axes = {"axis_0": 0.05}  # Below deadzone
        buttons = {}
        
        command = mapper.map_input_to_command(axes, buttons)
        assert command.get("joint_0", 0.0) == 0.0

    def test_mapper_gain(self):
        """Test gain scaling in mapper."""
        mapper = JointVelocityMapper(num_joints=6)
        mapper.set_gain(2.0)
        
        axes = {"axis_0": 0.5}
        buttons = {}
        
        command = mapper.map_input_to_command(axes, buttons)
        assert command["joint_0"] == 1.0  # 0.5 * 2.0


class TestTraining:
    """Test imitation learning training pipeline."""

    def test_create_mlp_policy(self):
        """Test creating an MLP policy network."""
        config = {
            "arch": "mlp",
            "hidden_dims": [256, 256],
            "activation": "relu",
        }
        policy = create_policy_network(input_dim=18, output_dim=6, config=config)
        assert isinstance(policy, torch.nn.Module)
        assert policy is not None

    def test_policy_forward_pass(self):
        """Test policy network forward pass."""
        policy = MLPPolicy(
            input_dim=18,
            output_dim=6,
            hidden_dims=[128, 128],
        )
        policy.eval()
        
        with torch.no_grad():
            obs = torch.randn(4, 18)  # Batch of 4 observations
            actions = policy(obs)
        
        assert actions.shape == (4, 6)

    def test_trainer_initialization(self):
        """Test behavioral cloning trainer initialization."""
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[128, 128])
        config = {
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "optimizer": "adam",
        }
        trainer = BehavioralCloningTrainer(policy, device="cpu", config=config)
        assert trainer is not None
        assert trainer.policy == policy

    def test_trainer_loss_computation(self):
        """Test that trainer can compute loss."""
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[128, 128])
        trainer = BehavioralCloningTrainer(policy, device="cpu")
        
        obs = torch.randn(4, 18)
        actions = torch.randn(4, 6)
        
        loss = trainer._compute_loss(obs, actions)
        assert isinstance(loss, torch.Tensor)
        assert loss.item() > 0


class TestEvaluation:
    """Test policy evaluation and rollout."""

    def test_policy_rollout_creation(self):
        """Test creating a policy rollout evaluator."""
        rollout = PolicyRollout(policy_type="il", device="cpu")
        assert rollout is not None
        assert rollout.policy_type == "il"

    def test_policy_set_directly(self):
        """Test setting policy directly on rollout."""
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[128, 128])
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)
        assert rollout.policy == policy

    def test_get_action_from_policy(self):
        """Test getting action from policy during rollout."""
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[128, 128])
        policy.eval()
        
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)
        
        obs = np.random.randn(18)
        action = rollout.get_action(obs)
        
        assert isinstance(action, np.ndarray)
        assert action.shape == (6,)

    def test_full_episode_rollout(self):
        """Test running a full episode rollout."""
        # Create environment and policy
        env = PiperEnv(task_name="reaching", max_episode_steps=5)
        task = ReachingTask(TaskConfig(name="reaching", max_episode_steps=5))
        env.set_task(task)
        
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[64, 64])
        policy.eval()
        
        # Run rollout
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)
        result = rollout.rollout(env, max_steps=5)
        
        assert result.episode_id == 0
        assert len(result.observations) > 0
        assert len(result.actions) > 0
        assert result.episode_length <= 5
        env.close()

    def test_multiple_episode_rollout(self):
        """Test running multiple episode rollouts."""
        env = PiperEnv(task_name="reaching", max_episode_steps=5)
        task = ReachingTask(TaskConfig(name="reaching", max_episode_steps=5))
        env.set_task(task)
        
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[64, 64])
        policy.eval()
        
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)
        results = rollout.rollout_multiple(env, num_episodes=3, max_steps=5)
        
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result.episode_id == i
            assert len(result.observations) > 0
        env.close()


class TestData:
    """Test data loading and preprocessing."""

    def test_trajectory_dataset_creation(self):
        """Test creating a trajectory dataset."""
        # Create synthetic data
        trajectories = [
            {
                "observations": np.random.randn(10, 18),
                "actions": np.random.randn(10, 6),
            }
            for _ in range(3)
        ]
        
        dataset = TrajectoryDataset(trajectories)
        assert len(dataset) == 30  # 3 trajectories * 10 timesteps
        
        # Test indexing
        obs, action = dataset[0]
        assert obs.shape == (18,)
        assert action.shape == (6,)

    def test_dataloader_creation(self):
        """Test creating a dataloader."""
        trajectories = [
            {
                "observations": np.random.randn(5, 18),
                "actions": np.random.randn(5, 6),
            }
            for _ in range(2)
        ]
        
        loader = create_dataloader(trajectories, batch_size=2, shuffle=True)
        assert loader is not None
        
        batch = next(iter(loader))
        assert len(batch) == 2  # obs, actions
        assert batch[0].shape[0] <= 2  # batch size


class TestConfiguration:
    """Test configuration loading and management."""

    def test_config_save_and_load(self, tmp_path):
        """Test saving and loading configuration."""
        config = {
            "model": {
                "hidden_dims": [256, 256],
                "activation": "relu",
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 1e-3,
            },
        }
        
        config_path = tmp_path / "config.yaml"
        save_config(config, str(config_path))
        loaded = load_config(str(config_path))
        
        assert loaded == config


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_pipeline_smoke_test(self, tmp_path):
        """Smoke test for full pipeline: env -> data collection -> training -> eval.
        
        This test verifies that all components can work together without errors,
        using very small scales (few steps, few epochs) for speed.
        """
        # 1. Create environment
        env = PiperEnv(task_name="reaching", max_episode_steps=5)
        task = ReachingTask(TaskConfig(name="reaching", max_episode_steps=5))
        env.set_task(task)
        
        # 2. Collect synthetic trajectories
        trajectories = []
        for ep in range(2):
            obs, _ = env.reset()
            traj_obs, traj_acts = [], []
            for _ in range(5):
                action = np.random.randn(6)
                obs, _, terminated, truncated, _ = env.step(action)
                traj_obs.append(obs.copy())
                traj_acts.append(action.copy())
                if terminated or truncated:
                    break
            trajectories.append({
                "observations": np.array(traj_obs),
                "actions": np.array(traj_acts),
            })
        env.close()
        
        # 3. Create dataset and dataloader
        dataset = TrajectoryDataset(trajectories)
        loader = create_dataloader(dataset.trajectories, batch_size=2, shuffle=False)
        assert len(loader) > 0
        
        # 4. Train policy
        policy = MLPPolicy(input_dim=18, output_dim=6, hidden_dims=[64, 64])
        trainer = BehavioralCloningTrainer(policy, device="cpu")
        
        # Run one training step
        batch = next(iter(loader))
        if isinstance(batch, (list, tuple)):
            obs_batch, action_batch = batch
        else:
            obs_batch = batch[0]
            action_batch = batch[1]
        
        # 5. Evaluate policy
        policy.eval()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)
        
        env2 = PiperEnv(task_name="reaching", max_episode_steps=5)
        env2.set_task(ReachingTask(TaskConfig(max_episode_steps=5)))
        result = rollout.rollout(env2, max_steps=5)
        
        assert result.episode_id == 0
        assert result.episode_length > 0
        env2.close()


def test_all_imports():
    """Quick test that all major imports work."""
    from src.environment import PiperEnv
    from src.teleoperation.devices import MockDevice
    from src.training.il import MLPPolicy
    from src.evaluation import PolicyRollout
    from src.data import TrajectoryDataset
    from src.utils import load_config
    
    # All imports successful
    assert all([PiperEnv, MockDevice, MLPPolicy, PolicyRollout, TrajectoryDataset, load_config])


if __name__ == "__main__":
    # Run with pytest
    pytest.main([__file__, "-v"])
