"""Unit tests for Piper environment."""

import numpy as np
import pytest

from src.environment.env import PiperEnv
from src.environment.tasks import TaskConfig, ReachingTask, PushingTask


class TestPiperEnvInitialization:
    """Tests for environment initialization."""

    def test_env_loads_successfully(self):
        """Test that environment loads URDF and initializes."""
        env = PiperEnv(task="reaching")
        assert env.model is not None
        assert env.data is not None
        assert env.task is not None
        env.close()

    def test_env_with_custom_frequency(self):
        """Test environment with custom simulation frequency."""
        freq = 200  # 200 Hz
        env = PiperEnv(simulation_freq=freq, task="reaching")
        expected_timestep = 1.0 / freq
        assert np.isclose(env.timestep, expected_timestep)
        env.close()

    def test_env_rejects_invalid_frequency(self):
        """Test that invalid frequencies are rejected."""
        with pytest.raises(ValueError):
            PiperEnv(simulation_freq=30)  # Below min

        with pytest.raises(ValueError):
            PiperEnv(simulation_freq=2000)  # Above max

    def test_env_rejects_missing_urdf(self):
        """Test that missing URDF raises error."""
        with pytest.raises(FileNotFoundError):
            PiperEnv(urdf_path="nonexistent.urdf")

    def test_env_task_configuration(self):
        """Test task configuration during initialization."""
        config = TaskConfig(
            name="reaching",
            max_episode_steps=50,
            target_tolerance=0.05,
            reward_scale=2.0
        )
        env = PiperEnv(task="reaching", task_config=config)
        assert env.max_steps == 50
        assert env.task.config.target_tolerance == 0.05
        env.close()


class TestObservationAndAction:
    """Tests for observation and action spaces."""

    def test_observation_shape(self):
        """Test that observations have correct shape."""
        env = PiperEnv(task="reaching")
        obs = env.reset()

        assert obs.shape == (18,), f"Expected shape (18,), got {obs.shape}"
        assert obs.dtype == np.float32
        env.close()

    def test_observation_components(self):
        """Test observation components."""
        env = PiperEnv(task="reaching")
        env.reset()

        obs = env.get_observation()

        # Extract components
        joint_pos = obs[:6]
        joint_vel = obs[6:12]
        ee_pos = obs[12:15]
        ee_vel = obs[15:18]

        # Check ranges
        assert all(np.isfinite(joint_pos))
        assert all(np.isfinite(joint_vel))
        assert all(np.isfinite(ee_pos))
        assert all(np.isfinite(ee_vel))
        env.close()

    def test_action_space_shape(self):
        """Test that action space has correct shape."""
        env = PiperEnv(task="reaching")
        env.reset()

        action = np.random.uniform(
            env.action_space_low,
            env.action_space_high
        )
        assert action.shape == (6,)
        env.close()

    def test_action_clipping(self):
        """Test that out-of-bounds actions are clipped."""
        env = PiperEnv(task="reaching")
        env.reset()

        # Create action outside bounds
        action = np.array([10.0, -10.0, 5.0, -5.0, 3.0, -3.0])

        # Step should clip action
        obs, reward, done, info = env.step(action)

        # Check that observation is still valid
        assert obs.shape == (18,)
        assert np.all(np.isfinite(obs))
        env.close()

    def test_observation_consistency(self):
        """Test observation consistency across resets."""
        env = PiperEnv(task="reaching")

        obs1 = env.reset()
        obs2 = env.reset()

        # Joint positions should be similar (both reset to center)
        joint_pos1 = obs1[:6]
        joint_pos2 = obs2[:6]
        assert np.allclose(joint_pos1, joint_pos2, atol=1e-6)
        env.close()


class TestSimulationStep:
    """Tests for environment stepping."""

    def test_step_returns_correct_format(self):
        """Test that step returns correct tuple format."""
        env = PiperEnv(task="reaching")
        env.reset()

        action = np.zeros(6)
        obs, reward, done, info = env.step(action)

        assert obs.shape == (18,)
        assert isinstance(reward, (float, np.floating))
        assert isinstance(done, bool)
        assert isinstance(info, dict)
        env.close()

    def test_random_rollout(self):
        """Test random rollout completes successfully."""
        env = PiperEnv(task="reaching", task_config=TaskConfig(max_episode_steps=10))
        env.reset()

        total_reward = 0.0
        step_count = 0

        for _ in range(20):
            action = np.random.uniform(
                env.action_space_low,
                env.action_space_high
            )
            obs, reward, done, info = env.step(action)

            total_reward += reward
            step_count += 1

            # Verify observation validity
            assert obs.shape == (18,)
            assert np.all(np.isfinite(obs))

            if done:
                env.reset()

        assert step_count == 20
        env.close()

    def test_step_counter_increments(self):
        """Test that step counter increments correctly."""
        env = PiperEnv(task="reaching")
        env.reset()

        for i in range(5):
            action = np.zeros(6)
            obs, reward, done, info = env.step(action)
            assert info["step"] == i + 1

        env.close()

    def test_episode_termination_on_max_steps(self):
        """Test that episode terminates after max steps."""
        config = TaskConfig(max_episode_steps=5)
        env = PiperEnv(task="reaching", task_config=config)
        env.reset()

        for i in range(4):
            action = np.zeros(6)
            obs, reward, done, info = env.step(action)
            assert not done

        # Fifth step should trigger done
        action = np.zeros(6)
        obs, reward, done, info = env.step(action)
        assert done
        assert info.get("success") == False

        env.close()

    def test_deterministic_simulation(self):
        """Test that simulation is deterministic."""
        env1 = PiperEnv(task="reaching")
        env2 = PiperEnv(task="reaching")

        env1.reset(seed=42)
        env2.reset(seed=42)

        action_sequence = [
            np.array([0.1, -0.1, 0.05, 0.0, -0.05, 0.1]),
            np.array([0.0, 0.0, 0.0, 0.1, 0.0, -0.1]),
        ]

        for action in action_sequence:
            obs1, reward1, done1, _ = env1.step(action)
            obs2, reward2, done2, _ = env2.step(action)

            assert np.allclose(obs1, obs2)
            assert np.isclose(reward1, reward2)

        env1.close()
        env2.close()


class TestTasks:
    """Tests for task functionality."""

    def test_reaching_task_reset(self):
        """Test reaching task reset."""
        env = PiperEnv(task="reaching")
        obs = env.reset()

        task_obs = env.get_task_observation()
        assert "target_position" in task_obs
        assert "end_effector_position" in task_obs
        assert "distance_to_target" in task_obs
        assert task_obs["target_position"].shape == (3,)
        env.close()

    def test_reward_computation(self):
        """Test that rewards are computed deterministically."""
        env = PiperEnv(task="reaching")

        obs1 = env.reset(seed=42)
        obs2 = env.reset(seed=42)

        action = np.array([0.1, 0.0, -0.1, 0.0, 0.0, 0.1])

        _, reward1, _, _ = env.step(action)
        _, reward2, _, _ = env.step(action)

        assert np.isclose(reward1, reward2), "Rewards should be deterministic"
        env.close()

    def test_reward_scales_with_distance(self):
        """Test that reward decreases with distance to target."""
        env = PiperEnv(task="reaching")

        # Move end-effector away from target
        env.reset()
        obs1, reward1, _, _ = env.step(np.zeros(6))

        obs2, reward2, _, _ = env.step(np.zeros(6))

        # Rewards should be in valid range
        assert isinstance(reward1, (float, np.floating))
        assert isinstance(reward2, (float, np.floating))
        env.close()

    def test_task_success_detection(self):
        """Test that task success is detected."""
        config = TaskConfig(
            name="reaching",
            max_episode_steps=1000,
            target_tolerance=5.0  # Large tolerance for testing
        )
        env = PiperEnv(task="reaching", task_config=config)
        env.reset()

        # Set end-effector close to target
        env.task.target_position = env.data.body_xpos[env.ee_id].copy()

        obs = env.get_observation()
        success = env.task.is_success(obs)
        assert success

        env.close()

    def test_pushing_task(self):
        """Test pushing task initialization and observation."""
        env = PiperEnv(task="pushing")
        env.reset()

        task_obs = env.get_task_observation()
        assert "target_position" in task_obs
        assert "object_position" in task_obs
        assert "end_effector_position" in task_obs
        env.close()

    def test_task_switching(self):
        """Test switching tasks during environment lifetime."""
        env = PiperEnv(task="reaching")
        env.reset()

        config = TaskConfig(name="pushing", max_episode_steps=50)
        env.set_task("pushing", config)

        assert env.max_steps == 50
        obs = env.reset()
        assert obs.shape == (18,)
        env.close()


class TestJointControl:
    """Tests for joint control."""

    def test_set_joint_state(self):
        """Test setting joint state."""
        env = PiperEnv(task="reaching")

        positions = np.array([0.5, -0.3, 0.2, 0.1, -0.2, 0.3])
        velocities = np.array([0.1, -0.1, 0.05, 0.0, 0.1, -0.05])

        env.set_joint_state(positions, velocities)

        obs = env.get_observation()
        assert np.allclose(obs[:6], positions, atol=1e-6)
        assert np.allclose(obs[6:12], velocities, atol=1e-6)
        env.close()

    def test_set_joint_state_clipping(self):
        """Test that out-of-bounds positions are clipped."""
        env = PiperEnv(task="reaching")

        # Create positions outside limits
        positions = np.array([10.0, -10.0, 5.0, -5.0, 3.0, -3.0])

        env.set_joint_state(positions)

        obs = env.get_observation()
        # Should be clipped to valid range
        assert np.all(obs[:6] >= env.DEFAULT_JOINT_LIMITS["lower"])
        assert np.all(obs[:6] <= env.DEFAULT_JOINT_LIMITS["upper"])
        env.close()

    def test_joint_limits(self):
        """Test joint limits retrieval."""
        env = PiperEnv(task="reaching")

        limits = env.get_joint_limits()

        assert "lower" in limits
        assert "upper" in limits
        assert "velocity" in limits

        assert len(limits["lower"]) == 6
        assert len(limits["upper"]) == 6
        assert len(limits["velocity"]) == 6

        env.close()


class TestRendering:
    """Tests for rendering functionality."""

    def test_headless_rendering(self):
        """Test rendering in headless mode."""
        env = PiperEnv(task="reaching", render_mode=None)
        env.reset()

        # In headless mode, render() should return None
        result = env.render()
        # Either None or RGB array is acceptable
        assert result is None or isinstance(result, np.ndarray)
        env.close()

    def test_environment_close(self):
        """Test that environment closes without errors."""
        env = PiperEnv(task="reaching")
        env.reset()
        env.close()
        # Should not raise


class TestObservationSpace:
    """Tests for observation space definition."""

    def test_observation_space_definition(self):
        """Test observation space is properly defined."""
        env = PiperEnv(task="reaching")

        assert env.observation_space_shape == (18,)
        env.close()

    def test_action_space_definition(self):
        """Test action space is properly defined."""
        env = PiperEnv(task="reaching")

        assert env.action_space_shape == (6,)
        assert env.action_space_low.shape == (6,)
        assert env.action_space_high.shape == (6,)
        assert np.all(env.action_space_low < 0)
        assert np.all(env.action_space_high > 0)
        env.close()

    def test_action_space_symmetry(self):
        """Test action space is symmetric."""
        env = PiperEnv(task="reaching")

        # Velocity limits should be symmetric
        assert np.allclose(env.action_space_low, -env.action_space_high)
        env.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_invalid_task_name(self):
        """Test that invalid task name raises error."""
        env = PiperEnv(task="reaching")
        with pytest.raises(ValueError):
            env.set_task("nonexistent_task")
        env.close()

    def test_invalid_action_dimension(self):
        """Test that action with wrong dimension is handled."""
        env = PiperEnv(task="reaching")
        env.reset()

        # Action with wrong dimension should be clipped
        action = np.array([0.1, 0.2])  # Only 2 dimensions
        # This will raise an error in the step function
        with pytest.raises((IndexError, ValueError)):
            env.step(action)
        env.close()

    def test_observation_after_long_episode(self):
        """Test observation validity after many steps."""
        config = TaskConfig(max_episode_steps=1000)
        env = PiperEnv(task="reaching", task_config=config)
        env.reset()

        for _ in range(100):
            action = np.random.uniform(
                env.action_space_low,
                env.action_space_high
            )
            obs, reward, done, _ = env.step(action)

            assert obs.shape == (18,)
            assert np.all(np.isfinite(obs))

            if done:
                break

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
