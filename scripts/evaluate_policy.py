#!/usr/bin/env python3
"""Script for evaluating trained policies.

Usage:
    python scripts/evaluate_policy.py --checkpoint-path path/to/checkpoint.pt \
        --env-config config/env.yaml --task reaching --num-episodes 100
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.evaluation.rollout import PolicyRollout
from src.evaluation.metrics import EvaluationMetrics
from src.evaluation.compare import PolicyComparator

logger = logging.getLogger(__name__)


def setup_logging(output_dir: Optional[Path] = None) -> None:
    """Setup logging configuration.

    Args:
        output_dir: Directory for log files
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
    )

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / "evaluation.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter(log_format))
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Log file: {log_file}")


def load_environment(env_config: Dict[str, Any]) -> Any:
    """Load environment from config.

    Args:
        env_config: Environment configuration dictionary

    Returns:
        Environment instance

    Note:
        This is a placeholder. Actual implementation depends on environment
        module structure.
    """
    # Import environment module (to be implemented)
    # from src.environment.env import PiperReachingEnv
    #
    # env_type = env_config.get("type", "reaching")
    # if env_type == "reaching":
    #     return PiperReachingEnv(**env_config.get("params", {}))
    # else:
    #     raise ValueError(f"Unknown environment type: {env_type}")

    logger.warning("Environment module not yet implemented. Using mock environment.")
    # Return mock environment for now
    return MockEnvironment()


def load_config(config_path: Path) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML is invalid
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)

    return config or {}


def evaluate_single_policy(
    checkpoint_path: Path,
    env: Any,
    num_episodes: int = 100,
    max_steps: int = 1000,
    task_name: str = "reaching",
    policy_type: str = "il",
    device: str = "cpu",
    output_dir: Optional[Path] = None,
    render: bool = False,
) -> Dict[str, Any]:
    """Evaluate a single policy.

    Args:
        checkpoint_path: Path to policy checkpoint
        env: Environment instance
        num_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        task_name: Name of task
        policy_type: Type of policy ("il" or "rl")
        device: Device for policy ("cpu" or "cuda")
        output_dir: Directory to save results
        render: Whether to render episodes

    Returns:
        Dictionary with evaluation results
    """
    logger.info(f"Evaluating policy: {checkpoint_path}")
    logger.info(f"  Policy type: {policy_type}")
    logger.info(f"  Episodes: {num_episodes}")
    logger.info(f"  Max steps: {max_steps}")

    # Create rollout and evaluate
    rollout = PolicyRollout(
        checkpoint_path=checkpoint_path,
        policy_type=policy_type,
        device=device,
        deterministic=True,
    )

    results = rollout.rollout_multiple(
        env,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render=render,
    )

    # Compute metrics
    metrics_computer = EvaluationMetrics(task_name)
    metrics = metrics_computer.compute_metrics(results)

    logger.info(f"\nEvaluation Results for {checkpoint_path.name}:")
    logger.info(str(metrics))

    # Compute smoothness metrics
    smoothness = metrics_computer.compute_trajectory_smoothness(results)
    if smoothness:
        logger.info("\nTrajectory Smoothness:")
        for key, value in smoothness.items():
            logger.info(f"  {key}: {value:.4f}")

    # Save results
    results_dict = {
        "checkpoint_path": str(checkpoint_path),
        "num_episodes": num_episodes,
        "max_steps": max_steps,
        "metrics": metrics.to_dict(),
        "smoothness": smoothness,
        "episode_results": [
            {
                "episode_id": r.episode_id,
                "return": r.episode_return,
                "length": r.episode_length,
                "success": r.success,
                "task_info": r.task_info,
            }
            for r in results
        ],
    }

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Results saved to {results_file}")

    return results_dict


def compare_policies(
    checkpoint_paths: Dict[str, Path],
    env: Any,
    num_episodes: int = 100,
    max_steps: int = 1000,
    task_name: str = "reaching",
    policy_type: str = "il",
    device: str = "cpu",
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Compare multiple policies.

    Args:
        checkpoint_paths: Dictionary mapping policy names to checkpoint paths
        env: Environment instance
        num_episodes: Number of evaluation episodes per policy
        max_steps: Maximum steps per episode
        task_name: Name of task
        policy_type: Type of policy
        device: Device for policies
        output_dir: Directory to save results

    Returns:
        Dictionary with comparison results
    """
    logger.info(f"Comparing {len(checkpoint_paths)} policies")

    # Load policies
    policies = {}
    for name, path in checkpoint_paths.items():
        policies[name] = PolicyRollout(
            checkpoint_path=path,
            policy_type=policy_type,
            device=device,
            deterministic=True,
        )

    # Compare
    comparator = PolicyComparator(task_name)
    comparison = comparator.compare_policies(
        env,
        policies,
        num_episodes=num_episodes,
        max_steps=max_steps,
    )

    logger.info(str(comparison))

    # Save results
    results_dict = comparison.to_dict()

    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        results_file = output_dir / "comparison_results.json"
        with open(results_file, "w") as f:
            json.dump(results_dict, f, indent=2)
        logger.info(f"Comparison results saved to {results_file}")

    return results_dict


def main() -> None:
    """Main evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate trained policies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate single IL policy
  python scripts/evaluate_policy.py \\
    --checkpoint-path checkpoints/il_model.pt \\
    --num-episodes 100 \\
    --task reaching

  # Compare two policies
  python scripts/evaluate_policy.py \\
    --compare il_checkpoint.pt rl_checkpoint.pt \\
    --policy-names IL RL \\
    --num-episodes 50
        """,
    )

    # Arguments for single policy evaluation
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        help="Path to policy checkpoint for single evaluation",
    )

    # Arguments for policy comparison
    parser.add_argument(
        "--compare",
        nargs="+",
        type=Path,
        help="Paths to policies for comparison",
    )
    parser.add_argument(
        "--policy-names",
        nargs="+",
        help="Names of policies for comparison",
    )

    # Common arguments
    parser.add_argument(
        "--env-config",
        type=Path,
        default=Path("config/env.yaml"),
        help="Path to environment config",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=100,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=1000,
        help="Maximum steps per episode",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reaching",
        choices=["reaching", "grasping"],
        help="Task type",
    )
    parser.add_argument(
        "--policy-type",
        type=str,
        default="il",
        choices=["il", "rl"],
        help="Type of policy",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/evaluation"),
        help="Output directory for results",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.output_dir)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("Starting policy evaluation")

    # Load environment
    try:
        if args.env_config.exists():
            env_config = load_config(args.env_config)
            env = load_environment(env_config)
        else:
            logger.warning(f"Env config not found: {args.env_config}, using mock environment")
            env = MockEnvironment()
    except Exception as e:
        logger.error(f"Failed to load environment: {e}")
        sys.exit(1)

    # Evaluate or compare
    try:
        if args.checkpoint_path:
            # Single policy evaluation
            evaluate_single_policy(
                checkpoint_path=args.checkpoint_path,
                env=env,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                task_name=args.task,
                policy_type=args.policy_type,
                device=args.device,
                output_dir=args.output_dir,
                render=args.render,
            )
        elif args.compare:
            # Policy comparison
            if not args.policy_names:
                args.policy_names = [p.stem for p in args.compare]

            checkpoint_dict = dict(zip(args.policy_names, args.compare))
            compare_policies(
                checkpoint_paths=checkpoint_dict,
                env=env,
                num_episodes=args.num_episodes,
                max_steps=args.max_steps,
                task_name=args.task,
                policy_type=args.policy_type,
                device=args.device,
                output_dir=args.output_dir,
            )
        else:
            parser.print_help()
            sys.exit(0)

        logger.info("Evaluation complete")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        sys.exit(1)


class MockEnvironment:
    """Mock environment for testing evaluation framework."""

    def __init__(self):
        """Initialize mock environment."""
        self.obs_dim = 10
        self.action_dim = 6
        self.step_count = 0
        self.max_steps = 100

    def reset(self):
        """Reset environment."""
        self.step_count = 0
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        return obs, {"task": "reaching"}

    def step(self, action):
        """Step environment."""
        obs = np.random.randn(self.obs_dim).astype(np.float32)
        reward = np.random.randn()
        self.step_count += 1
        done = self.step_count >= self.max_steps
        truncated = False
        info = {
            "task": "reaching",
            "success": np.random.rand() > 0.5,
            "task_info": {
                "final_distance": np.random.rand(),
                "end_effector_error": np.random.rand(),
            },
        }
        return obs, reward, done, truncated, info

    def render(self):
        """Render environment."""
        pass


if __name__ == "__main__":
    main()
