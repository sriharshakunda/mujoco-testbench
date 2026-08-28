"""Task definitions for Piper arm manipulation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Tuple

import numpy as np


@dataclass
class TaskConfig:
    """Configuration for manipulation tasks.

    Attributes:
        name: Task identifier (e.g., "reaching", "grasping")
        max_episode_steps: Maximum steps per episode
        target_tolerance: Distance tolerance for task success (meters)
        reward_scale: Scaling factor for reward computation
        success_reward: Reward given when task is succeeded
        failure_reward: Reward given when task fails
        step_penalty: Penalty applied each step (negative reward)
    """
    name: str = "reaching"
    max_episode_steps: int = 100
    target_tolerance: float = 0.02  # 2cm tolerance for reaching
    reward_scale: float = 1.0
    success_reward: float = 1.0
    failure_reward: float = 0.0
    step_penalty: float = 0.01
    custom_params: Dict = field(default_factory=dict)


class Task(ABC):
    """Abstract base class for manipulation tasks."""

    def __init__(self, config: TaskConfig):
        """Initialize task.

        Args:
            config: Task configuration
        """
        self.config = config
        self.target = None

    @abstractmethod
    def reset(self, observation: np.ndarray) -> None:
        """Reset task state.

        Args:
            observation: Current robot observation
        """
        pass

    @abstractmethod
    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        done: bool
    ) -> Tuple[float, Dict]:
        """Compute reward for current step.

        Args:
            observation: Current robot observation
            action: Action taken
            done: Whether episode is done

        Returns:
            reward: Scalar reward value
            info: Dictionary with reward details
        """
        pass

    @abstractmethod
    def is_success(self, observation: np.ndarray) -> bool:
        """Check if task is succeeded.

        Args:
            observation: Current robot observation

        Returns:
            True if task succeeded
        """
        pass

    @abstractmethod
    def get_task_observation(self, observation: np.ndarray) -> Dict:
        """Get task-specific observation.

        Args:
            observation: Current robot observation

        Returns:
            Dictionary with task-specific state
        """
        pass


class ReachingTask(Task):
    """Reaching task: move end-effector to target position.

    The task is to move the end-effector to a target position in 3D space.
    Reward is based on distance to target, with bonus for reaching it.
    """

    def __init__(self, config: TaskConfig):
        """Initialize reaching task.

        Args:
            config: Task configuration
        """
        super().__init__(config)
        self.target_position = np.array([0.3, 0.0, 0.3])  # Default target
        self.max_distance = 1.0  # Max distance for reward shaping

    def reset(self, observation: np.ndarray) -> None:
        """Reset task by sampling new target position.

        The target is sampled within the workspace of the arm.

        Args:
            observation: Current robot observation
        """
        # Sample target position in reachable workspace
        # Workspace: x in [0.2, 0.8], y in [-0.3, 0.3], z in [0.0, 0.6]
        self.target_position = np.array([
            np.random.uniform(0.2, 0.8),
            np.random.uniform(-0.3, 0.3),
            np.random.uniform(0.0, 0.6)
        ])

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        done: bool
    ) -> Tuple[float, Dict]:
        """Compute reaching reward.

        Reward is based on:
        1. Distance to target (negative distance-based reward)
        2. Success bonus (1.0 if within tolerance)
        3. Step penalty

        Args:
            observation: Current robot observation with shape (6, 2)
            action: Joint velocity action
            done: Whether episode is done

        Returns:
            reward: Scalar reward
            info: Dictionary with reward components
        """
        # Extract end-effector position from observation
        # Observation format: [q1, q2, q3, q4, q5, q6, x_ee, y_ee, z_ee, ...]
        ee_position = observation[-3:] if len(observation) >= 9 else np.array([0.0, 0.0, 0.0])

        # Compute distance to target
        distance = np.linalg.norm(ee_position - self.target_position)

        # Distance-based reward (normalized by max_distance)
        distance_reward = -distance / self.max_distance

        # Success reward if within tolerance
        success_bonus = 0.0
        if distance <= self.config.target_tolerance:
            success_bonus = self.config.success_reward

        # Total reward with step penalty
        reward = (distance_reward + success_bonus - self.config.step_penalty)
        reward *= self.config.reward_scale

        info = {
            "distance_to_target": float(distance),
            "distance_reward": float(distance_reward),
            "success_bonus": float(success_bonus),
            "step_penalty": float(self.config.step_penalty),
            "target_position": self.target_position.copy(),
            "ee_position": ee_position.copy(),
        }

        return reward, info

    def is_success(self, observation: np.ndarray) -> bool:
        """Check if task is succeeded.

        Args:
            observation: Current robot observation

        Returns:
            True if end-effector is within tolerance of target
        """
        ee_position = observation[-3:] if len(observation) >= 9 else np.array([0.0, 0.0, 0.0])
        distance = np.linalg.norm(ee_position - self.target_position)
        return distance <= self.config.target_tolerance

    def get_task_observation(self, observation: np.ndarray) -> Dict:
        """Get task-specific observation.

        Args:
            observation: Current robot observation

        Returns:
            Dictionary with task state
        """
        # Extract joint positions and velocities
        joint_positions = observation[:6] if len(observation) >= 6 else np.zeros(6)
        joint_velocities = observation[6:12] if len(observation) >= 12 else np.zeros(6)
        ee_position = observation[12:15] if len(observation) >= 15 else np.zeros(3)
        ee_velocity = observation[15:18] if len(observation) >= 18 else np.zeros(3)

        return {
            "target_position": self.target_position.copy(),
            "end_effector_position": ee_position.copy(),
            "end_effector_velocity": ee_velocity.copy(),
            "joint_positions": joint_positions.copy(),
            "joint_velocities": joint_velocities.copy(),
            "distance_to_target": float(np.linalg.norm(ee_position - self.target_position)),
        }


class PushingTask(Task):
    """Pushing task: push an object to a target position.

    This task requires the end-effector to push an object to a target.
    """

    def __init__(self, config: TaskConfig):
        """Initialize pushing task.

        Args:
            config: Task configuration
        """
        super().__init__(config)
        self.target_position = np.array([0.4, 0.0, 0.01])
        self.object_position = np.array([0.3, 0.0, 0.01])
        self.max_distance = 1.0

    def reset(self, observation: np.ndarray) -> None:
        """Reset pushing task.

        Args:
            observation: Current robot observation
        """
        # Reset object and target positions
        self.object_position = np.array([
            np.random.uniform(0.2, 0.5),
            np.random.uniform(-0.2, 0.2),
            0.01
        ])
        self.target_position = np.array([
            self.object_position[0] + np.random.uniform(0.1, 0.3),
            self.object_position[1] + np.random.uniform(-0.1, 0.1),
            0.01
        ])

    def compute_reward(
        self,
        observation: np.ndarray,
        action: np.ndarray,
        done: bool
    ) -> Tuple[float, Dict]:
        """Compute pushing reward.

        Args:
            observation: Current robot observation
            action: Joint velocity action
            done: Whether episode is done

        Returns:
            reward: Scalar reward
            info: Dictionary with reward components
        """
        ee_position = observation[-3:] if len(observation) >= 9 else np.array([0.0, 0.0, 0.0])

        # Distance from object to target
        object_distance = np.linalg.norm(self.object_position - self.target_position)

        # Distance from ee to object (for pushing)
        ee_to_object = np.linalg.norm(ee_position - self.object_position)

        # Reward based on object position to target
        distance_reward = -object_distance / self.max_distance

        # Penalty for ee not being close to object
        contact_penalty = 0.0
        if ee_to_object > 0.1:
            contact_penalty = -0.1

        # Success bonus
        success_bonus = 0.0
        if object_distance <= self.config.target_tolerance:
            success_bonus = self.config.success_reward

        reward = (distance_reward + contact_penalty + success_bonus - self.config.step_penalty)
        reward *= self.config.reward_scale

        info = {
            "object_distance_to_target": float(object_distance),
            "ee_to_object_distance": float(ee_to_object),
            "distance_reward": float(distance_reward),
            "contact_penalty": float(contact_penalty),
            "success_bonus": float(success_bonus),
        }

        return reward, info

    def is_success(self, observation: np.ndarray) -> bool:
        """Check if pushing task is succeeded.

        Args:
            observation: Current robot observation

        Returns:
            True if object is within tolerance of target
        """
        distance = np.linalg.norm(self.object_position - self.target_position)
        return distance <= self.config.target_tolerance

    def get_task_observation(self, observation: np.ndarray) -> Dict:
        """Get task-specific observation.

        Args:
            observation: Current robot observation

        Returns:
            Dictionary with task state
        """
        ee_position = observation[-3:] if len(observation) >= 9 else np.zeros(3)

        return {
            "target_position": self.target_position.copy(),
            "object_position": self.object_position.copy(),
            "end_effector_position": ee_position.copy(),
            "distance_to_target": float(np.linalg.norm(self.object_position - self.target_position)),
        }


def create_task(task_name: str, config: TaskConfig = None) -> Task:
    """Factory function to create task instances.

    Args:
        task_name: Name of task ("reaching", "pushing")
        config: Task configuration (uses defaults if None)

    Returns:
        Task instance

    Raises:
        ValueError: If task name is not recognized
    """
    if config is None:
        config = TaskConfig(name=task_name)

    if task_name == "reaching":
        return ReachingTask(config)
    elif task_name == "pushing":
        return PushingTask(config)
    else:
        raise ValueError(f"Unknown task: {task_name}")
