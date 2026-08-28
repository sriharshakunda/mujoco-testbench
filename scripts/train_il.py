#!/usr/bin/env python3
"""Behavioral cloning training script.

This script provides the main entry point for training an imitation learning
policy using behavioral cloning on demonstration data. It handles:
- Configuration loading
- Dataset creation and loading
- Model instantiation
- Training loop execution
- Checkpoint saving and evaluation
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import TrajectoryDataset, create_dataloader, create_train_val_split
from src.training import BehavioralCloningTrainer, create_policy_network
from src.utils import load_config, save_config, setup_logging


logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def create_datasets(config: dict) -> tuple:
    """Create training and validation datasets from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_dataset, val_dataset, dataset_stats)

    Raises:
        FileNotFoundError: If data files not found
        ValueError: If data loading fails
    """
    data_config = config.get("data", {})

    # Load data paths
    train_obs_path = data_config.get("train_data_path")
    train_actions_path = data_config.get("train_actions_path")

    if train_obs_path is None or train_actions_path is None:
        raise ValueError(
            "Must specify train_data_path and train_actions_path in config"
        )

    # Load or create dataset
    logger.info("Loading training data...")
    dataset = TrajectoryDataset(
        observations=train_obs_path,
        actions=train_actions_path,
    )

    # Get dataset statistics
    stats = dataset.get_stats()
    logger.info(f"Dataset stats: {stats}")

    # Create train/val split
    train_ratio = data_config.get("train_val_split", 0.8)
    seed = data_config.get("seed", 42)

    train_dataset, val_dataset = create_train_val_split(
        dataset,
        train_ratio=train_ratio,
        seed=seed,
    )

    return train_dataset, val_dataset, stats


def create_dataloaders(
    train_dataset,
    val_dataset,
    config: dict,
) -> tuple:
    """Create data loaders from datasets.

    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    data_config = config.get("data", {})

    batch_size = config.get("training", {}).get("batch_size", 32)
    num_workers = data_config.get("num_workers", 4)
    pin_memory = data_config.get("pin_memory", True)

    logger.info("Creating data loaders...")

    train_loader = create_dataloader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = create_dataloader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader


def main(
    config_path: str,
    override_lr: Optional[float] = None,
    override_epochs: Optional[int] = None,
) -> None:
    """Main training function.

    Args:
        config_path: Path to configuration YAML file
        override_lr: Optional override for learning rate
        override_epochs: Optional override for number of epochs
    """
    # Load configuration
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Apply command-line overrides
    if override_lr is not None:
        config["training"]["learning_rate"] = override_lr
        logger.info(f"Overriding learning rate to {override_lr}")

    if override_epochs is not None:
        config["training"]["num_epochs"] = override_epochs
        logger.info(f"Overriding num_epochs to {override_epochs}")

    # Set random seed for reproducibility
    seed = config.get("data", {}).get("seed", 42)
    logger.info(f"Setting random seed to {seed}")
    set_seed(seed)

    # Set deterministic algorithms if requested
    if config.get("advanced", {}).get("deterministic", True):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        logger.info("Enabled deterministic training")

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset, dataset_stats = create_datasets(config)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        train_dataset,
        val_dataset,
        config,
    )

    # Determine device
    device_name = config.get("training", {}).get("device", "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, using CPU")
        device_name = "cpu"

    logger.info(f"Using device: {device_name}")

    # Create model
    logger.info("Creating policy network...")
    obs_dim = dataset_stats["obs_dim"]
    action_dim = dataset_stats["action_dim"]

    model = create_policy_network(
        input_dim=obs_dim,
        output_dim=action_dim,
        config=config.get("model", {}),
    )

    logger.info(
        f"Created model with obs_dim={obs_dim}, action_dim={action_dim}"
    )
    logger.info(f"Model:\n{model}")

    # Create trainer
    logger.info("Creating trainer...")
    trainer = BehavioralCloningTrainer(
        model=model,
        device=device_name,
        config=config.get("training", {}),
    )

    # Load checkpoint if resuming
    checkpoint_config = config.get("checkpoint", {})
    resume_path = checkpoint_config.get("resume_from")
    if resume_path is not None:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        trainer.load_checkpoint(
            Path(resume_path),
            resume_training=checkpoint_config.get("resume_training", False),
        )

    # Run training
    logger.info("Starting training...")
    num_epochs = config.get("training", {}).get("num_epochs", 100)
    val_interval = config.get("training", {}).get("val_interval", 1)
    checkpoint_interval = config.get("training", {}).get("checkpoint_interval", 1)

    training_stats = trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=num_epochs,
        val_interval=val_interval,
        checkpoint_interval=checkpoint_interval,
    )

    # Save training configuration to checkpoint directory
    config_save_path = trainer.checkpoint_dir / "config.yaml"
    save_config(config, str(config_save_path))
    logger.info(f"Configuration saved to {config_save_path}")

    # Log final results
    logger.info("=" * 60)
    logger.info("Training Complete")
    logger.info("=" * 60)
    logger.info(f"Best validation loss: {training_stats['best_val_loss']:.6f}")
    logger.info(f"Best epoch: {training_stats['best_epoch'] + 1}")
    logger.info(f"Checkpoint directory: {trainer.checkpoint_dir}")
    logger.info("=" * 60)

    # Optional: evaluate on test set (if separate test data exists)
    test_config = config.get("data", {})
    if (
        test_config.get("test_data_path") is not None
        and test_config.get("test_actions_path") is not None
    ):
        logger.info("Evaluating on test set...")
        test_dataset = TrajectoryDataset(
            observations=test_config.get("test_data_path"),
            actions=test_config.get("test_actions_path"),
        )
        test_loader = create_dataloader(
            test_dataset,
            batch_size=config.get("training", {}).get("batch_size", 32),
            shuffle=False,
            num_workers=0,
        )

        # Load best model
        best_checkpoint = trainer.checkpoint_dir / "best_model.pt"
        if best_checkpoint.exists():
            trainer.load_checkpoint(best_checkpoint, resume_training=False)

        test_loss = trainer.evaluate(test_loader)
        logger.info(f"Test set loss: {test_loss:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train imitation learning policy using behavioral cloning"
    )
    parser.add_argument(
        "config",
        type=str,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate from config",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs from config",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level",
    )

    args = parser.parse_args()

    # Set up logging
    log_level = getattr(logging, args.log_level)
    setup_logging(level=log_level)

    try:
        main(
            config_path=args.config,
            override_lr=args.learning_rate,
            override_epochs=args.num_epochs,
        )
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        sys.exit(1)
