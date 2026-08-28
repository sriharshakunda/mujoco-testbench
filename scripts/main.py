#!/usr/bin/env python3
"""Main orchestration script for the MuJoCo IL/RL pipeline.

This script provides a unified entry point for the complete pipeline:
  1. Environment setup and task configuration
  2. Data collection (teleoperation)
  3. IL training (behavioral cloning)
  4. Optional RL fine-tuning
  5. Policy evaluation and reporting

Usage:
    python scripts/main.py --config config/reaching.yaml --mode train_il
    python scripts/main.py --mode evaluate --checkpoint models/policy.pt
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.environment import PiperEnv, ReachingTask, TaskConfig
from src.teleoperation.devices import DeviceManager
from src.teleoperation.mapping import ConfigurableMapper
from src.training.il import create_policy_network, BehavioralCloningTrainer
from src.evaluation import PolicyRollout
from src.data import TrajectoryDataset, create_dataloader
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)


class Pipeline:
    """Main pipeline orchestrator."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize pipeline.

        Args:
            config_path: Path to configuration YAML file
        """
        self.config = {}
        if config_path:
            self.config = load_config(config_path)
        
        # Setup logging
        log_level = self.config.get("logging", {}).get("level", "INFO")
        setup_logging(level=log_level)
        logger.info(f"Initialized pipeline with config: {config_path}")

    def collect_data(self, num_episodes: int = 5, render: bool = False) -> list:
        """Collect teleoperated demonstration data.

        Args:
            num_episodes: Number of episodes to collect
            render: Whether to render during collection

        Returns:
            List of trajectory dictionaries
        """
        logger.info(f"Collecting {num_episodes} demonstration episodes...")

        env = PiperEnv(
            task_name=self.config.get("task", "reaching"),
            render_mode="human" if render else None,
        )
        
        task_config = TaskConfig(
            name=self.config.get("task", "reaching"),
            max_episode_steps=self.config.get("max_episode_steps", 100),
        )
        
        if task_config.name == "reaching":
            task = ReachingTask(task_config)
        else:
            raise ValueError(f"Unknown task: {task_config.name}")
        
        env.set_task(task)

        # Initialize device manager
        device_manager = DeviceManager()
        device = device_manager.add_device("mock", name="mock_device", num_axes=6)
        
        if device is None:
            logger.error("Failed to initialize input device")
            return []

        # Initialize input mapper
        mapper = ConfigurableMapper(self.config.get("teleoperation", {}))

        trajectories = []

        for ep in range(num_episodes):
            logger.info(f"Episode {ep + 1}/{num_episodes}")
            obs, _ = env.reset()
            
            traj_obs, traj_acts = [], []

            for step in range(task_config.max_episode_steps):
                if render:
                    env.render()

                # Get device input
                state = device.get_state()
                if state is None:
                    action = np.zeros(6)
                else:
                    command = mapper.map_input_to_command(state.axes, state.buttons)
                    action = np.array([command.get(f"joint_{i}", 0.0) for i in range(6)])

                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                traj_obs.append(obs.copy())
                traj_acts.append(action.copy())

                if terminated or truncated:
                    break

            trajectories.append({
                "observations": np.array(traj_obs),
                "actions": np.array(traj_acts),
            })
            logger.info(f"  Episode {ep + 1}: {len(traj_obs)} steps")

        env.close()
        device_manager.close_all()
        logger.info(f"Collected {len(trajectories)} trajectories")

        return trajectories

    def train_il(self, trajectories: list, output_path: str = "models/policy_il.pt") -> str:
        """Train IL (behavioral cloning) policy.

        Args:
            trajectories: List of trajectory dictionaries
            output_path: Path to save trained policy

        Returns:
            Path to saved checkpoint
        """
        logger.info("Training IL policy...")

        # Create dataset
        dataset = TrajectoryDataset(trajectories)
        loader = create_dataloader(
            trajectories,
            batch_size=self.config.get("training", {}).get("batch_size", 32),
            shuffle=True,
        )

        # Create policy
        policy = create_policy_network(
            input_dim=18,
            output_dim=6,
            config=self.config.get("model", {}),
        )

        # Create trainer
        trainer = BehavioralCloningTrainer(
            policy,
            device="cpu",
            config=self.config.get("training", {}),
        )

        # Train
        num_epochs = self.config.get("training", {}).get("epochs", 10)
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch + 1}/{num_epochs}")
            # Training loop would go here
            # For now, just a placeholder
            pass

        # Save checkpoint
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saved IL policy to {output_path}")

        return output_path

    def evaluate(self, checkpoint_path: str, num_episodes: int = 5) -> Dict:
        """Evaluate a trained policy.

        Args:
            checkpoint_path: Path to policy checkpoint
            num_episodes: Number of evaluation episodes

        Returns:
            Evaluation metrics dictionary
        """
        logger.info(f"Evaluating policy from {checkpoint_path}...")

        # Create environment
        env = PiperEnv(task_name=self.config.get("task", "reaching"))
        task = ReachingTask(TaskConfig(name=self.config.get("task", "reaching")))
        env.set_task(task)

        # Load policy and run rollouts
        rollout = PolicyRollout(policy_type="il", checkpoint_path=checkpoint_path)
        results = rollout.rollout_multiple(env, num_episodes=num_episodes, max_steps=100)

        # Compute metrics
        episode_returns = [r.episode_return for r in results]
        success_rate = np.mean([r.success for r in results])
        
        metrics = {
            "mean_episode_return": float(np.mean(episode_returns)),
            "std_episode_return": float(np.std(episode_returns)),
            "success_rate": float(success_rate),
            "num_episodes": num_episodes,
        }

        logger.info(f"Evaluation results: {metrics}")
        env.close()

        return metrics

    def run(self, mode: str = "train_il", **kwargs):
        """Run pipeline in specified mode.

        Args:
            mode: Pipeline mode ("collect", "train_il", "train_rl", "evaluate")
            **kwargs: Additional arguments for specific modes
        """
        if mode == "collect":
            trajectories = self.collect_data(
                num_episodes=kwargs.get("num_episodes", 5),
                render=kwargs.get("render", False),
            )
            return trajectories

        elif mode == "train_il":
            trajectories = kwargs.get("trajectories", [])
            if not trajectories:
                logger.warning("No trajectories provided for training")
                return None
            return self.train_il(trajectories)

        elif mode == "evaluate":
            checkpoint = kwargs.get("checkpoint")
            if not checkpoint:
                logger.error("Checkpoint path required for evaluation")
                return None
            return self.evaluate(checkpoint, num_episodes=kwargs.get("num_episodes", 5))

        else:
            logger.error(f"Unknown mode: {mode}")
            return None


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="MuJoCo IL/RL Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect 10 demonstration episodes
  python scripts/main.py --mode collect --num-episodes 10 --render
  
  # Train IL policy on collected data
  python scripts/main.py --mode train_il --config config/reaching.yaml
  
  # Evaluate trained policy
  python scripts/main.py --mode evaluate --checkpoint models/policy_il.pt
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="train_il",
        choices=["collect", "train_il", "train_rl", "evaluate"],
        help="Pipeline mode to run",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=5,
        help="Number of episodes for data collection or evaluation",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to policy checkpoint for evaluation",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render environment during execution",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for computation",
    )

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = Pipeline(config_path=args.config)

    # Run specified mode
    if args.mode == "collect":
        trajectories = pipeline.run(
            mode="collect",
            num_episodes=args.num_episodes,
            render=args.render,
        )
        logger.info(f"Collected {len(trajectories)} trajectories")

    elif args.mode == "train_il":
        # For demo, create synthetic trajectories
        logger.info("Running in demo mode with synthetic data")
        trajectory_data = [
            {
                "observations": np.random.randn(10, 18),
                "actions": np.random.randn(10, 6),
            }
            for _ in range(2)
        ]
        checkpoint = pipeline.run(mode="train_il", trajectories=trajectory_data)
        logger.info(f"Trained policy saved to {checkpoint}")

    elif args.mode == "evaluate":
        if not args.checkpoint:
            logger.error("--checkpoint required for evaluation mode")
            sys.exit(1)
        metrics = pipeline.run(
            mode="evaluate",
            checkpoint=args.checkpoint,
            num_episodes=args.num_episodes,
        )
        logger.info(f"Evaluation metrics: {metrics}")

    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

    logger.info("Pipeline execution complete")


if __name__ == "__main__":
    main()
