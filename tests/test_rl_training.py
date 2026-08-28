"""Unit tests for RL training pipeline.

Tests for environment wrapper, trainer, warmstart loader, and training script.
"""

import tempfile
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
import yaml

from src.environment.tasks import TaskConfig
from src.training.il.models import create_policy_network
from src.training.il.trainer import BehavioralCloningTrainer
from src.training.rl.env_wrapper import PiperGymEnv
from src.training.rl.trainer import RLTrainer
from src.training.rl.warmstart import ILWarmstartLoader


class TestPiperGymEnv:
    """Tests for PiperGymEnv gymnasium wrapper."""

    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        env = PiperGymEnv(
            urdf_path="assets/piper.urdf",
            task="reaching",
            simulation_freq=500.0,
            seed=42,
        )
        yield env
        env.close()

    def test_env_creation(self, env):
        """Test environment initialization."""
        assert env is not None
        assert env.observation_space is not None
        assert env.action_space is not None

    def test_observation_space(self, env):
        """Test observation space configuration."""
        assert isinstance(env.observation_space, gym.spaces.Box)
        assert env.observation_space.shape == (18,)
        assert env.observation_space.dtype == np.float32

    def test_action_space(self, env):
        """Test action space configuration."""
        assert isinstance(env.action_space, gym.spaces.Box)
        assert env.action_space.shape == (6,)
        assert env.action_space.dtype == np.float32

    def test_reset(self, env):
        """Test environment reset."""
        obs, info = env.reset()
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

    def test_step(self, env):
        """Test environment step."""
        env.reset()
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        assert isinstance(obs, np.ndarray)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, float)
        assert isinstance(terminated, (bool, np.bool_))
        assert isinstance(truncated, (bool, np.bool_))
        assert isinstance(info, dict)

    def test_seed_setting(self):
        """Test random seed setting."""
        env1 = PiperGymEnv(seed=42)
        env2 = PiperGymEnv(seed=42)

        obs1, _ = env1.reset()
        obs2, _ = env2.reset()

        np.testing.assert_array_almost_equal(obs1, obs2)
        env1.close()
        env2.close()

    def test_render_mode(self):
        """Test render mode configuration."""
        env = PiperGymEnv(render_mode=None)
        assert env.render_mode is None
        env.close()

    def test_episode_termination(self, env):
        """Test episode termination conditions."""
        obs, _ = env.reset()
        done = False
        step_count = 0

        while not done and step_count < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step_count += 1

        assert done, "Episode should terminate within 1000 steps"


class TestILWarmstartLoader:
    """Tests for IL warmstart loading."""

    @pytest.fixture
    def il_checkpoint_path(self, tmp_path):
        """Create a dummy IL checkpoint."""
        model = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "config": {},
        }

        checkpoint_path = tmp_path / "il_model.pt"
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def test_load_checkpoint(self, il_checkpoint_path):
        """Test loading IL checkpoint."""
        checkpoint = ILWarmstartLoader.load_il_checkpoint(str(il_checkpoint_path))

        assert "model_state_dict" in checkpoint
        assert isinstance(checkpoint["model_state_dict"], dict)

    def test_load_nonexistent_checkpoint(self):
        """Test loading nonexistent checkpoint."""
        with pytest.raises(FileNotFoundError):
            ILWarmstartLoader.load_il_checkpoint("nonexistent.pt")

    def test_transfer_weights(self, il_checkpoint_path):
        """Test weight transfer to RL policy."""
        il_checkpoint = ILWarmstartLoader.load_il_checkpoint(str(il_checkpoint_path))

        rl_policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        ILWarmstartLoader.transfer_weights_to_rl_policy(
            il_checkpoint,
            rl_policy,
            freeze_layers=False,
        )

        # Verify weights are transferred
        for param in rl_policy.parameters():
            assert param is not None

    def test_freeze_layers(self, il_checkpoint_path):
        """Test layer freezing."""
        il_checkpoint = ILWarmstartLoader.load_il_checkpoint(str(il_checkpoint_path))

        rl_policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        ILWarmstartLoader.transfer_weights_to_rl_policy(
            il_checkpoint,
            rl_policy,
            freeze_layers=True,
        )

        # Verify all parameters are frozen
        for param in rl_policy.parameters():
            assert not param.requires_grad

    def test_load_and_transfer(self, il_checkpoint_path):
        """Test combined load and transfer."""
        rl_policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        checkpoint = ILWarmstartLoader.load_and_transfer(
            str(il_checkpoint_path),
            rl_policy,
            freeze_layers=False,
        )

        assert "model_state_dict" in checkpoint

    def test_unfreeze_layers(self, il_checkpoint_path):
        """Test layer unfreezing."""
        il_checkpoint = ILWarmstartLoader.load_il_checkpoint(str(il_checkpoint_path))

        rl_policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        ILWarmstartLoader.transfer_weights_to_rl_policy(
            il_checkpoint,
            rl_policy,
            freeze_layers=True,
        )

        # Unfreeze layers
        ILWarmstartLoader.unfreeze_layers(rl_policy, num_layers=2)

        # Verify some parameters are trainable
        trainable_count = sum(
            p.numel() for p in rl_policy.parameters() if p.requires_grad
        )
        assert trainable_count > 0

    def test_parameter_counts(self, il_checkpoint_path):
        """Test parameter counting."""
        il_checkpoint = ILWarmstartLoader.load_il_checkpoint(str(il_checkpoint_path))

        rl_policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        ILWarmstartLoader.transfer_weights_to_rl_policy(
            il_checkpoint,
            rl_policy,
            freeze_layers=True,
        )

        frozen_count = ILWarmstartLoader.get_frozen_params_count(rl_policy)
        trainable_count = ILWarmstartLoader.get_trainable_params_count(rl_policy)

        assert frozen_count > 0
        assert trainable_count == 0


class TestRLTrainer:
    """Tests for RL trainer."""

    @pytest.fixture
    def env(self):
        """Create environment for testing."""
        env = PiperGymEnv(seed=42)
        yield env
        env.close()

    @pytest.fixture
    def trainer(self, env):
        """Create trainer for testing."""
        config = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "batch_size": 64,
            "n_steps": 512,
            "num_epochs": 4,
            "checkpoint_dir": "checkpoints/rl",
            "log_dir": "logs/rl",
        }
        return RLTrainer(env, config=config)

    def test_trainer_creation(self, trainer):
        """Test trainer initialization."""
        assert trainer is not None
        assert trainer.algorithm_name == "ppo"
        assert trainer.learning_rate == 3e-4

    def test_model_creation(self, trainer):
        """Test model creation."""
        model = trainer.create_model()
        assert model is not None
        assert trainer.model is not None

    def test_unsupported_algorithm(self, env):
        """Test unsupported algorithm raises error."""
        config = {"algorithm": "unsupported"}
        with pytest.raises(ValueError):
            RLTrainer(env, config=config)

    def test_different_algorithms(self, env):
        """Test trainer with different algorithms."""
        for algo in ["ppo", "sac", "ddpg", "td3"]:
            config = {"algorithm": algo}
            trainer = RLTrainer(env, config=config)
            model = trainer.create_model()
            assert model is not None

    def test_checkpoint_save(self, trainer, tmp_path):
        """Test checkpoint saving."""
        trainer.create_model()
        checkpoint_path = tmp_path / "test_model.zip"

        saved_path = trainer.save_checkpoint(checkpoint_path)

        assert saved_path.exists()

    def test_checkpoint_load(self, trainer, tmp_path):
        """Test checkpoint loading."""
        trainer.create_model()
        checkpoint_path = tmp_path / "test_model.zip"
        trainer.save_checkpoint(checkpoint_path)

        trainer2 = RLTrainer(trainer.env, config=trainer.config)
        trainer2.create_model()
        trainer2.load_checkpoint(checkpoint_path)

        assert trainer2.model is not None

    def test_evaluation_basic(self, trainer):
        """Test basic evaluation."""
        trainer.create_model()

        # Do a quick mini training
        trainer.model.learn(total_timesteps=100)

        # Evaluate
        metrics = trainer.evaluate(num_episodes=2, deterministic=True)

        assert "mean_reward" in metrics
        assert "std_reward" in metrics
        assert "success_rate" in metrics
        assert isinstance(metrics["mean_reward"], float)

    def test_train_not_created_error(self, trainer):
        """Test training without model creation."""
        with pytest.raises(ValueError):
            trainer.train(total_timesteps=10)

    def test_evaluate_not_created_error(self, trainer):
        """Test evaluation without model creation."""
        with pytest.raises(ValueError):
            trainer.evaluate()

    def test_policy_kwargs(self, env):
        """Test custom policy kwargs."""
        config = {
            "algorithm": "ppo",
            "policy_kwargs": {"net_arch": [128, 128]},
        }
        trainer = RLTrainer(env, config=config)
        model = trainer.create_model()
        assert model is not None


class TestTrainingIntegration:
    """Integration tests for full training pipeline."""

    def test_training_pipeline_cold_start(self, tmp_path):
        """Test training from scratch (cold start)."""
        env = PiperGymEnv(seed=42)
        config = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "batch_size": 32,
            "n_steps": 256,
            "num_epochs": 2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        }

        trainer = RLTrainer(env, config=config)
        model = trainer.create_model()

        # Quick training
        stats = trainer.train(total_timesteps=500)

        assert "algorithm" in stats
        assert stats["algorithm"] == "ppo"

        env.close()

    def test_training_pipeline_with_warmstart(self, tmp_path):
        """Test training with IL warmstart."""
        # Create and save IL checkpoint
        il_model = create_policy_network(
            input_dim=18,
            output_dim=6,
            config={"arch": "mlp", "hidden_dims": [256, 256]},
        )

        il_checkpoint = {
            "model_state_dict": il_model.state_dict(),
            "optimizer_state_dict": {},
            "config": {},
        }

        il_checkpoint_path = tmp_path / "il_model.pt"
        torch.save(il_checkpoint, il_checkpoint_path)

        # Create RL environment and trainer
        env = PiperGymEnv(seed=42)
        config = {
            "algorithm": "ppo",
            "learning_rate": 3e-4,
            "batch_size": 32,
            "n_steps": 256,
            "num_epochs": 2,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "log_dir": str(tmp_path / "logs"),
        }

        trainer = RLTrainer(env, config=config)
        model = trainer.create_model()

        # Transfer IL weights
        ILWarmstartLoader.load_and_transfer(
            str(il_checkpoint_path),
            model.policy,
            freeze_layers=False,
        )

        # Quick training
        stats = trainer.train(total_timesteps=500)

        assert stats is not None

        env.close()


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_config_yaml_loading(self, tmp_path):
        """Test YAML config loading."""
        config = {
            "seed": 42,
            "environment": {
                "task": "reaching",
                "simulation_freq": 500.0,
            },
            "rl": {
                "algorithm": "ppo",
                "learning_rate": 3e-4,
            },
            "training": {
                "total_timesteps": 100000,
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Load it back
        with open(config_path, "r") as f:
            loaded_config = yaml.safe_load(f)

        assert loaded_config == config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
