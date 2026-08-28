#!/usr/bin/env python3
"""RL training script for Piper arm manipulation.

Loads configuration, creates environment and trainer, and trains RL policy
with optional IL warmstart initialization.

Example usage:
    python scripts/train_rl.py --config config/rl_config.yaml
    python scripts/train_rl.py --config config/rl_config.yaml --il-checkpoint checkpoints/il/best_model.pt
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml

from src.environment.tasks import TaskConfig
from src.training.il.models import create_policy_network
from src.training.rl.env_wrapper import PiperGymEnv
from src.training.rl.trainer import RLTrainer
from src.training.rl.warmstart import ILWarmstartLoader


def setup_logging(log_dir: Path) -> None:
    """Setup logging to file and console.

    Args:
        log_dir: Directory for log files
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "rl_training.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )

    logger = logging.getLogger(__name__)
    logger.info(f"Logging to {log_file}")


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file not found
        yaml.YAMLError: If YAML parsing fails
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def create_environment(config: dict, render: bool = False) -> PiperGymEnv:
    """Create training environment.

    Args:
        config: Environment configuration dictionary
        render: Whether to render environment

    Returns:
        Initialized environment

    Raises:
        ValueError: If configuration is invalid
    """
    env_config = config.get("environment", {})

    urdf_path = env_config.get("urdf_path", "assets/piper.urdf")
    task = env_config.get("task", "reaching")
    simulation_freq = env_config.get("simulation_freq", 500.0)
    seed = config.get("seed", None)

    # Create task config
    task_config_dict = env_config.get("task_config", {})
    task_config = TaskConfig(
        name=task,
        max_episode_steps=task_config_dict.get("max_episode_steps", 500),
        target_tolerance=task_config_dict.get("target_tolerance", 0.05),
        reward_scale=task_config_dict.get("reward_scale", 1.0),
        success_reward=task_config_dict.get("success_reward", 1.0),
        failure_reward=task_config_dict.get("failure_reward", 0.0),
        step_penalty=task_config_dict.get("step_penalty", 0.01),
    )

    env = PiperGymEnv(
        urdf_path=urdf_path,
        task=task,
        task_config=task_config,
        simulation_freq=simulation_freq,
        render_mode="human" if render else None,
        seed=seed,
    )

    return env


def load_il_checkpoint_to_rl(
    il_checkpoint_path: str,
    rl_policy,
    freeze_layers: bool = False,
    device: str = "cpu",
) -> None:
    """Load IL checkpoint weights to RL policy.

    Args:
        il_checkpoint_path: Path to IL checkpoint
        rl_policy: RL policy network
        freeze_layers: Whether to freeze transferred layers
        device: Device to transfer to

    Raises:
        FileNotFoundError: If checkpoint not found
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Loading IL checkpoint from {il_checkpoint_path}")

    checkpoint = ILWarmstartLoader.load_il_checkpoint(
        il_checkpoint_path,
        device=device,
    )

    ILWarmstartLoader.transfer_weights_to_rl_policy(
        checkpoint,
        rl_policy,
        freeze_layers=freeze_layers,
        device=device,
    )

    trainable = ILWarmstartLoader.get_trainable_params_count(rl_policy)
    frozen = ILWarmstartLoader.get_frozen_params_count(rl_policy)
    logger.info(f"Transferred IL weights. Trainable: {trainable}, Frozen: {frozen}")


def train_rl(
    config_path: str,
    il_checkpoint_path: Optional[str] = None,
    freeze_il_layers: bool = False,
    device: Optional[str] = None,
    render: bool = False,
    resume_checkpoint: Optional[str] = None,
) -> dict:
    """Train RL policy with optional IL warmstart.

    Args:
        config_path: Path to RL training configuration
        il_checkpoint_path: Optional path to IL checkpoint for warmstart
        freeze_il_layers: Whether to freeze IL layers
        device: Device to train on ("cpu" or "cuda")
        render: Whether to render environment
        resume_checkpoint: Path to checkpoint to resume from

    Returns:
        Training results dictionary

    Raises:
        FileNotFoundError: If config or checkpoint not found
        ValueError: If configuration is invalid
    """
    # Setup
    config = load_config(config_path)
    log_dir = Path(config.get("log_dir", "logs/rl"))
    setup_logging(log_dir)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Starting RL Training")
    logger.info("=" * 80)
    logger.info(f"Config: {json.dumps(config, indent=2, default=str)}")

    # Determine device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    # Set random seeds
    seed = config.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    logger.info(f"Set random seed: {seed}")

    # Create environment
    logger.info("Creating environment...")
    env = create_environment(config, render=render)
    logger.info(f"Environment created. Observation shape: {env.observation_space.shape}, "
                f"Action shape: {env.action_space.shape}")

    # Create RL trainer
    logger.info("Creating RL trainer...")
    rl_config = config.get("rl", {})
    trainer = RLTrainer(env, device=device, config=rl_config)

    # Create model
    logger.info("Creating RL model...")
    model = trainer.create_model()

    # Load IL checkpoint if provided
    if il_checkpoint_path:
        logger.info(f"Loading IL checkpoint: {il_checkpoint_path}")
        try:
            load_il_checkpoint_to_rl(
                il_checkpoint_path,
                model.policy,
                freeze_layers=freeze_il_layers,
                device=device,
            )
            logger.info("IL warmstart loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load IL checkpoint: {e}")
            raise

    # Resume from checkpoint if provided
    if resume_checkpoint:
        logger.info(f"Resuming from checkpoint: {resume_checkpoint}")
        try:
            trainer.load_checkpoint(Path(resume_checkpoint), resume_training=True)
            logger.info("Checkpoint loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            raise

    # Train
    logger.info("Starting training loop...")
    training_config = config.get("training", {})
    total_timesteps = training_config.get("total_timesteps", 100000)
    save_interval = training_config.get("save_interval", 10000)
    eval_interval = training_config.get("eval_interval", 5000)
    eval_episodes = training_config.get("eval_episodes", 10)

    try:
        stats = trainer.train(
            total_timesteps=total_timesteps,
            save_interval=save_interval,
            eval_interval=eval_interval,
            eval_episodes=eval_episodes,
        )

        logger.info("Training completed successfully")

    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        stats = {"status": "interrupted"}

    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

    # Final evaluation
    logger.info("Running final evaluation...")
    try:
        eval_metrics = trainer.evaluate(
            num_episodes=eval_episodes,
            deterministic=True,
        )
        stats.update({
            "final_evaluation": eval_metrics,
        })

        logger.info(f"Final evaluation metrics: {eval_metrics}")

    except Exception as e:
        logger.warning(f"Final evaluation failed: {e}")

    # Save results
    results_path = log_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(stats, f, indent=2, default=str)
    logger.info(f"Results saved to {results_path}")

    logger.info("=" * 80)
    logger.info("RL Training Complete")
    logger.info("=" * 80)

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train RL policy for Piper arm manipulation"
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to RL training configuration file (YAML)",
    )

    parser.add_argument(
        "--il-checkpoint",
        type=str,
        default=None,
        help="Path to IL checkpoint for warmstart",
    )

    parser.add_argument(
        "--freeze-il-layers",
        action="store_true",
        help="Freeze IL transferred layers during RL training",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to train on (cpu, cuda, cuda:0, etc.)",
    )

    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during training",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    # Train
    results = train_rl(
        config_path=args.config,
        il_checkpoint_path=args.il_checkpoint,
        freeze_il_layers=args.freeze_il_layers,
        device=args.device,
        render=args.render,
        resume_checkpoint=args.resume,
    )

    return results


if __name__ == "__main__":
    main()
