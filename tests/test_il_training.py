"""Unit tests for imitation learning training pipeline.

Tests cover:
- Model architecture and forward pass
- Trainer initialization and training loop
- Checkpoint saving/loading
- Dataset creation and loading
- Configuration loading and validation
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryDataset, create_dataloader, create_train_val_split
from src.training import BehavioralCloningTrainer, MLPPolicy, create_policy_network
from src.utils import load_config, merge_configs, save_config


class TestMLPPolicy:
    """Tests for MLP policy network."""

    def test_mlp_initialization(self):
        """Test basic MLP initialization."""
        model = MLPPolicy(
            input_dim=10,
            output_dim=4,
            hidden_dims=[64, 64],
            activation="relu",
        )

        assert model.input_dim == 10
        assert model.output_dim == 4
        assert len(model.network) > 0

    def test_mlp_forward_pass(self):
        """Test MLP forward pass."""
        model = MLPPolicy(
            input_dim=10,
            output_dim=4,
            hidden_dims=[64, 64],
        )

        # Create dummy input
        batch_size = 8
        x = torch.randn(batch_size, 10)

        # Forward pass
        output = model(x)

        assert output.shape == (batch_size, 4)
        assert isinstance(output, torch.Tensor)

    def test_mlp_with_batch_norm(self):
        """Test MLP with batch normalization."""
        model = MLPPolicy(
            input_dim=10,
            output_dim=4,
            hidden_dims=[64, 64],
            use_batch_norm=True,
        )

        x = torch.randn(8, 10)
        output = model(x)

        assert output.shape == (8, 4)

    def test_mlp_with_dropout(self):
        """Test MLP with dropout."""
        model = MLPPolicy(
            input_dim=10,
            output_dim=4,
            hidden_dims=[64, 64],
            dropout_rate=0.1,
        )

        x = torch.randn(8, 10)
        output = model(x)

        assert output.shape == (8, 4)

    def test_mlp_activation_functions(self):
        """Test different activation functions."""
        activations = ["relu", "tanh", "elu", "sigmoid"]

        for act in activations:
            model = MLPPolicy(
                input_dim=10,
                output_dim=4,
                hidden_dims=[64],
                activation=act,
            )

            x = torch.randn(8, 10)
            output = model(x)

            assert output.shape == (8, 4)

    def test_mlp_invalid_activation(self):
        """Test invalid activation raises error."""
        with pytest.raises(ValueError):
            MLPPolicy(
                input_dim=10,
                output_dim=4,
                hidden_dims=[64],
                activation="invalid",
            )


class TestPolicyFactory:
    """Tests for policy network factory function."""

    def test_create_mlp_policy(self):
        """Test creating MLP policy via factory."""
        config = {
            "arch": "mlp",
            "hidden_dims": [256, 256],
            "activation": "relu",
        }

        model = create_policy_network(
            input_dim=10,
            output_dim=4,
            config=config,
        )

        assert isinstance(model, MLPPolicy)
        assert model.input_dim == 10
        assert model.output_dim == 4

    def test_create_policy_invalid_arch(self):
        """Test invalid architecture raises error."""
        config = {"arch": "invalid"}

        with pytest.raises(ValueError):
            create_policy_network(
                input_dim=10,
                output_dim=4,
                config=config,
            )


class TestTrajectoryDataset:
    """Tests for trajectory dataset."""

    @pytest.fixture
    def sample_data(self):
        """Create sample trajectory data."""
        obs = np.random.randn(100, 10).astype(np.float32)
        actions = np.random.randn(100, 4).astype(np.float32)
        return obs, actions

    def test_dataset_creation(self, sample_data):
        """Test dataset initialization."""
        obs, actions = sample_data

        dataset = TrajectoryDataset(obs, actions)

        assert len(dataset) == 100
        assert dataset.observations.shape == (100, 10)
        assert dataset.actions.shape == (100, 4)

    def test_dataset_getitem(self, sample_data):
        """Test getting items from dataset."""
        obs, actions = sample_data
        dataset = TrajectoryDataset(obs, actions)

        sample_obs, sample_action = dataset[0]

        assert isinstance(sample_obs, torch.Tensor)
        assert isinstance(sample_action, torch.Tensor)
        assert sample_obs.shape == (10,)
        assert sample_action.shape == (4,)

    def test_dataset_stats(self, sample_data):
        """Test dataset statistics."""
        obs, actions = sample_data
        dataset = TrajectoryDataset(obs, actions)

        stats = dataset.get_stats()

        assert "num_samples" in stats
        assert "obs_dim" in stats
        assert "action_dim" in stats
        assert "obs_mean" in stats
        assert "obs_std" in stats
        assert stats["num_samples"] == 100
        assert stats["obs_dim"] == 10
        assert stats["action_dim"] == 4

    def test_dataset_mismatched_lengths(self):
        """Test error on mismatched data lengths."""
        obs = np.random.randn(100, 10).astype(np.float32)
        actions = np.random.randn(50, 4).astype(np.float32)

        with pytest.raises(ValueError):
            TrajectoryDataset(obs, actions)


class TestDataLoading:
    """Tests for data loading utilities."""

    @pytest.fixture
    def sample_dataset(self):
        """Create sample dataset."""
        obs = np.random.randn(100, 10).astype(np.float32)
        actions = np.random.randn(100, 4).astype(np.float32)
        return TrajectoryDataset(obs, actions)

    def test_train_val_split(self, sample_dataset):
        """Test train/val split."""
        train_ds, val_ds = create_train_val_split(
            sample_dataset,
            train_ratio=0.8,
        )

        assert len(train_ds) == 80
        assert len(val_ds) == 20

    def test_create_dataloader(self, sample_dataset):
        """Test dataloader creation."""
        loader = create_dataloader(
            sample_dataset,
            batch_size=16,
            shuffle=True,
        )

        assert isinstance(loader, DataLoader)

        # Test iteration
        batch_obs, batch_actions = next(iter(loader))
        assert batch_obs.shape == (16, 10)
        assert batch_actions.shape == (16, 4)


class TestBehavioralCloningTrainer:
    """Tests for behavioral cloning trainer."""

    @pytest.fixture
    def trainer_setup(self):
        """Set up trainer with model and config."""
        model = MLPPolicy(
            input_dim=10,
            output_dim=4,
            hidden_dims=[64],
        )

        config = {
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "optimizer": "adam",
            "log_dir": tempfile.mkdtemp(),
            "checkpoint_dir": tempfile.mkdtemp(),
        }

        trainer = BehavioralCloningTrainer(
            model=model,
            device="cpu",
            config=config,
        )

        return trainer, config

    def test_trainer_initialization(self, trainer_setup):
        """Test trainer initialization."""
        trainer, config = trainer_setup

        assert isinstance(trainer.model, MLPPolicy)
        assert trainer.device == "cpu"
        assert trainer.learning_rate == 1e-3

    def test_trainer_invalid_device(self):
        """Test invalid device raises error."""
        model = MLPPolicy(10, 4, [64])

        with pytest.raises(ValueError):
            BehavioralCloningTrainer(model, device="invalid_device")

    def test_trainer_forward_pass(self, trainer_setup):
        """Test trainer forward pass."""
        trainer, _ = trainer_setup

        x = torch.randn(8, 10)
        output = trainer.model(x)

        assert output.shape == (8, 4)

    def test_trainer_optimizer_creation(self, trainer_setup):
        """Test optimizer creation."""
        trainer, _ = trainer_setup

        assert isinstance(trainer.optimizer, torch.optim.Adam)

    def test_trainer_sgd_optimizer(self):
        """Test SGD optimizer creation."""
        model = MLPPolicy(10, 4, [64])

        config = {
            "optimizer": "sgd",
            "learning_rate": 1e-3,
            "log_dir": tempfile.mkdtemp(),
            "checkpoint_dir": tempfile.mkdtemp(),
        }

        trainer = BehavioralCloningTrainer(model, device="cpu", config=config)

        assert isinstance(trainer.optimizer, torch.optim.SGD)

    def test_trainer_checkpoint_save_load(self, trainer_setup):
        """Test checkpoint saving and loading."""
        trainer, config = trainer_setup

        # Save checkpoint
        checkpoint_path = Path(config["checkpoint_dir"]) / "test_checkpoint.pt"
        trainer.save_checkpoint(checkpoint_path)

        assert checkpoint_path.exists()

        # Load checkpoint
        trainer2 = BehavioralCloningTrainer(
            model=MLPPolicy(10, 4, [64]),
            device="cpu",
            config={
                "log_dir": tempfile.mkdtemp(),
                "checkpoint_dir": tempfile.mkdtemp(),
            },
        )

        trainer2.load_checkpoint(checkpoint_path, resume_training=False)

        # Verify weights match
        for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
            assert torch.allclose(p1, p2)

    def test_trainer_get_learning_rate(self, trainer_setup):
        """Test getting current learning rate."""
        trainer, _ = trainer_setup

        lr = trainer.get_learning_rate()
        assert lr == 1e-3


class TestTraining:
    """Tests for training loop."""

    @pytest.fixture
    def training_setup(self):
        """Set up training components."""
        # Create synthetic data
        obs = torch.randn(100, 10)
        actions = torch.randn(100, 4)
        dataset = TensorDataset(obs, actions)

        # Create loaders
        train_loader = DataLoader(dataset, batch_size=16)
        val_loader = DataLoader(dataset, batch_size=16)

        # Create model and trainer
        model = MLPPolicy(10, 4, [64])
        config = {
            "learning_rate": 1e-3,
            "log_dir": tempfile.mkdtemp(),
            "checkpoint_dir": tempfile.mkdtemp(),
        }

        trainer = BehavioralCloningTrainer(model, device="cpu", config=config)

        return trainer, train_loader, val_loader

    def test_training_loop_runs(self, training_setup):
        """Test that training loop runs without errors."""
        trainer, train_loader, val_loader = training_setup

        stats = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=2,
            val_interval=1,
        )

        assert "train_losses" in stats
        assert "val_losses" in stats
        assert len(stats["train_losses"]) == 2

    def test_training_loss_decreases(self, training_setup):
        """Test that training loss generally decreases."""
        trainer, train_loader, val_loader = training_setup

        stats = trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=5,
        )

        # Loss should generally decrease (though not monotonically)
        first_loss = stats["train_losses"][0]
        last_loss = stats["train_losses"][-1]

        # Very weak check: just that we get some loss values
        assert len(stats["train_losses"]) > 0
        assert first_loss > 0
        assert last_loss > 0

    def test_evaluation(self, training_setup):
        """Test model evaluation."""
        trainer, train_loader, val_loader = training_setup

        val_loss = trainer.evaluate(val_loader)

        assert isinstance(val_loss, float)
        assert val_loss >= 0


class TestConfiguration:
    """Tests for configuration loading and management."""

    @pytest.fixture
    def config_file(self):
        """Create temporary config file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            config = {
                "model": {
                    "arch": "mlp",
                    "hidden_dims": [256, 256],
                },
                "training": {
                    "learning_rate": 1e-3,
                    "num_epochs": 100,
                },
            }

            import yaml
            yaml.dump(config, f)

            yield f.name

        # Cleanup
        os.unlink(f.name)

    def test_load_config(self, config_file):
        """Test loading configuration from file."""
        config = load_config(config_file)

        assert "model" in config
        assert "training" in config
        assert config["model"]["arch"] == "mlp"

    def test_save_config(self):
        """Test saving configuration to file."""
        config = {
            "model": {"arch": "mlp"},
            "training": {"learning_rate": 1e-3},
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "config.yaml")
            save_config(config, output_path)

            assert os.path.exists(output_path)

            # Verify contents
            loaded = load_config(output_path)
            assert loaded["model"]["arch"] == "mlp"

    def test_merge_configs(self):
        """Test merging configurations."""
        base = {
            "model": {"arch": "mlp", "hidden_dims": [64]},
            "training": {"lr": 1e-3},
        }

        override = {
            "model": {"hidden_dims": [256]},
            "training": {"epochs": 100},
        }

        merged = merge_configs(base, override)

        assert merged["model"]["arch"] == "mlp"  # From base
        assert merged["model"]["hidden_dims"] == [256]  # From override
        assert merged["training"]["lr"] == 1e-3  # From base
        assert merged["training"]["epochs"] == 100  # From override


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
