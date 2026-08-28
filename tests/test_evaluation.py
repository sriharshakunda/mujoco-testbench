"""Unit tests for evaluation module."""

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.evaluation.compare import PolicyComparator, PolicyComparisonResult
from src.evaluation.metrics import EvaluationMetrics, MetricResult
from src.evaluation.rollout import PolicyRollout, RolloutResult


class SimplePolicy(nn.Module):
    """Simple policy network for testing."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 6):
        """Initialize simple policy."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim),
        )

    def forward(self, obs):
        """Forward pass."""
        return self.net(obs)


class MockEnvironment:
    """Mock environment for testing."""

    def __init__(self, obs_dim: int = 10, action_dim: int = 6):
        """Initialize mock environment."""
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.step_count = 0
        self.max_steps = 50
        self.viewer = None

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, {"task": "reaching"}

    def step(self, action):
        """Step environment."""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        # Simple reward: negative distance to zero action
        reward = -np.sum((action - 0) ** 2)
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False
        info = {
            "task": "reaching",
            "success": reward > -0.5,
            "task_info": {
                "final_distance": float(np.linalg.norm(action)),
                "end_effector_error": float(np.random.rand()),
            },
        }
        return obs, reward, done, truncated, info

    def render(self):
        """Render environment."""
        pass


class TestRolloutResult:
    """Test RolloutResult class."""

    def test_creation(self):
        """Test creating RolloutResult."""
        result = RolloutResult(
            episode_id=0,
            observations=[np.zeros((10,))],
            actions=[np.zeros((6,))],
            rewards=[1.0],
            dones=[True],
            episode_return=1.0,
            episode_length=1,
            success=True,
        )

        assert result.episode_id == 0
        assert len(result.observations) == 1
        assert result.episode_return == 1.0
        assert result.success

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = RolloutResult(
            episode_id=0,
            observations=[],
            actions=[],
            rewards=[],
            dones=[],
            episode_return=2.5,
            episode_length=10,
            success=True,
            task_info={"distance": 0.1},
        )

        d = result.to_dict()
        assert d["episode_id"] == 0
        assert d["episode_return"] == 2.5
        assert d["success"]


class TestPolicyRollout:
    """Test PolicyRollout class."""

    def test_initialization(self):
        """Test policy initialization."""
        rollout = PolicyRollout(policy_type="il", device="cpu")
        assert rollout.policy_type == "il"
        assert rollout.device == "cpu"

    def test_set_policy(self):
        """Test setting policy directly."""
        policy = SimplePolicy()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)

        assert rollout.policy is not None

    def test_get_action_il(self):
        """Test getting action from IL policy."""
        policy = SimplePolicy()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)

        obs = np.random.randn(10).astype(np.float32)
        action = rollout.get_action(obs)

        assert action.shape == (6,)
        assert isinstance(action, np.ndarray)

    def test_single_rollout(self):
        """Test running single episode."""
        policy = SimplePolicy()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)

        env = MockEnvironment()
        result = rollout.rollout(env, max_steps=50, episode_id=0)

        assert isinstance(result, RolloutResult)
        assert result.episode_id == 0
        assert len(result.observations) > 0
        assert len(result.actions) == len(result.rewards)

    def test_multiple_rollouts(self):
        """Test running multiple episodes."""
        policy = SimplePolicy()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)

        env = MockEnvironment()
        results = rollout.rollout_multiple(env, num_episodes=5, max_steps=50)

        assert len(results) == 5
        assert all(isinstance(r, RolloutResult) for r in results)

    def test_checkpoint_saving_loading(self):
        """Test saving and loading policy checkpoints."""
        policy = SimplePolicy()

        # Save checkpoint
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "policy.pt"
            torch.save(policy.state_dict(), checkpoint_path)

            # Create new rollout and load
            rollout = PolicyRollout(policy_type="il", device="cpu")
            rollout.load_policy(checkpoint_path)  # This should work if model is provided

            # For now, just test that checkpoint exists
            assert checkpoint_path.exists()


class TestMetrics:
    """Test metrics computation."""

    def test_metric_result_creation(self):
        """Test creating MetricResult."""
        metrics = MetricResult(
            mean_return=10.0,
            std_return=1.0,
            mean_length=50.0,
            std_length=5.0,
            success_rate=0.8,
            min_return=5.0,
            max_return=15.0,
            task_metrics={"distance": 0.5},
        )

        assert metrics.mean_return == 10.0
        assert metrics.success_rate == 0.8
        assert "distance" in metrics.task_metrics

    def test_compute_metrics(self):
        """Test computing metrics from rollout results."""
        # Create mock rollout results
        results = []
        for i in range(5):
            result = RolloutResult(
                episode_id=i,
                observations=[np.zeros((10,))],
                actions=[np.zeros((6,))],
                rewards=[float(i)],
                dones=[True],
                episode_return=float(i),
                episode_length=10,
                success=i > 2,
                task_info={"final_distance": float(i) * 0.1},
            )
            results.append(result)

        metrics_computer = EvaluationMetrics()
        metrics = metrics_computer.compute_metrics(results)

        assert isinstance(metrics, MetricResult)
        assert metrics.mean_return == 2.0  # Mean of 0, 1, 2, 3, 4
        assert metrics.success_rate == 0.4  # 2 out of 5
        assert metrics.mean_length == 10.0

    def test_reaching_metrics(self):
        """Test reaching-specific metrics."""
        results = []
        for i in range(3):
            result = RolloutResult(
                episode_id=i,
                observations=[],
                actions=[],
                rewards=[],
                dones=[],
                episode_return=0.0,
                episode_length=0,
                success=True,
                task_info={
                    "final_distance": 0.1 * i,
                    "end_effector_error": 0.05 * i,
                },
            )
            results.append(result)

        metrics_computer = EvaluationMetrics(task_name="reaching")
        metrics = metrics_computer.compute_metrics(results)

        assert "mean_final_distance" in metrics.task_metrics
        assert "mean_ee_error" in metrics.task_metrics

    def test_trajectory_smoothness(self):
        """Test trajectory smoothness metrics."""
        results = []
        for i in range(3):
            actions = [np.random.randn(6) for _ in range(20)]
            result = RolloutResult(
                episode_id=i,
                observations=[],
                actions=actions,
                rewards=[],
                dones=[],
                episode_return=0.0,
                episode_length=20,
                success=True,
            )
            results.append(result)

        metrics_computer = EvaluationMetrics()
        smoothness = metrics_computer.compute_trajectory_smoothness(results)

        assert "mean_action_variance" in smoothness
        assert "mean_jerk" in smoothness
        assert smoothness["mean_action_variance"] > 0

    def test_aggregate_metrics(self):
        """Test aggregating metrics across runs."""
        run1 = MetricResult(
            mean_return=10.0,
            std_return=1.0,
            mean_length=50.0,
            std_length=5.0,
            success_rate=0.8,
            min_return=5.0,
            max_return=15.0,
            task_metrics={},
        )

        run2 = MetricResult(
            mean_return=12.0,
            std_return=1.2,
            mean_length=55.0,
            std_length=6.0,
            success_rate=0.9,
            min_return=6.0,
            max_return=18.0,
            task_metrics={},
        )

        metrics_computer = EvaluationMetrics()
        aggregated = metrics_computer.aggregate_metrics([run1, run2])

        assert aggregated["mean_return"] == 11.0
        assert aggregated["mean_success_rate"] == 0.85


class TestPolicyComparator:
    """Test policy comparison."""

    def test_comparator_creation(self):
        """Test creating comparator."""
        comparator = PolicyComparator(task_name="reaching")
        assert comparator.task_name == "reaching"

    def test_compare_policies(self):
        """Test comparing two policies."""
        # Create two policies
        policy1 = SimplePolicy()
        policy2 = SimplePolicy()

        # Create rollouts
        rollout1 = PolicyRollout(policy_type="il", device="cpu")
        rollout1.set_policy(policy1)

        rollout2 = PolicyRollout(policy_type="il", device="cpu")
        rollout2.set_policy(policy2)

        # Create environment and comparator
        env = MockEnvironment()
        comparator = PolicyComparator(task_name="reaching")

        # Compare policies
        comparison = comparator.compare_policies(
            env=env,
            policies={"policy1": rollout1, "policy2": rollout2},
            num_episodes=5,
            max_steps=50,
        )

        assert isinstance(comparison, PolicyComparisonResult)
        assert len(comparison.policy_names) == 2
        assert comparison.best_policy in ["policy1", "policy2"]

    def test_comparison_result_to_dict(self):
        """Test converting comparison result to dict."""
        metrics1 = MetricResult(
            mean_return=10.0,
            std_return=1.0,
            mean_length=50.0,
            std_length=5.0,
            success_rate=0.8,
            min_return=5.0,
            max_return=15.0,
            task_metrics={},
        )

        comparison = PolicyComparisonResult(
            policy_names=["policy1"],
            metrics={"policy1": metrics1},
            statistical_significance={},
            best_policy="policy1",
            improvements={},
        )

        result_dict = comparison.to_dict()
        assert "policy_names" in result_dict
        assert "metrics" in result_dict
        assert "best_policy" in result_dict


class TestIntegration:
    """Integration tests."""

    def test_full_evaluation_pipeline(self):
        """Test full evaluation pipeline."""
        # Create policy
        policy = SimplePolicy()
        rollout = PolicyRollout(policy_type="il", device="cpu")
        rollout.set_policy(policy)

        # Create environment
        env = MockEnvironment()

        # Run evaluation
        results = rollout.rollout_multiple(env, num_episodes=3, max_steps=50)
        assert len(results) == 3

        # Compute metrics
        metrics_computer = EvaluationMetrics()
        metrics = metrics_computer.compute_metrics(results)
        assert metrics.mean_return is not None

        # Compute smoothness
        smoothness = metrics_computer.compute_trajectory_smoothness(results)
        assert len(smoothness) > 0

    def test_policy_comparison_pipeline(self):
        """Test policy comparison pipeline."""
        # Create two policies
        policy1 = SimplePolicy()
        policy2 = SimplePolicy()

        # Create rollouts
        rollout1 = PolicyRollout(policy_type="il", device="cpu")
        rollout1.set_policy(policy1)

        rollout2 = PolicyRollout(policy_type="il", device="cpu")
        rollout2.set_policy(policy2)

        # Compare
        env = MockEnvironment()
        comparator = PolicyComparator()

        comparison = comparator.compare_policies(
            env=env,
            policies={"policy1": rollout1, "policy2": rollout2},
            num_episodes=3,
            max_steps=50,
        )

        # Generate report
        report = comparator.generate_comparison_report(comparison)
        assert "policy1" in report
        assert "policy2" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
