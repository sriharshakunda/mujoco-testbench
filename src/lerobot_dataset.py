"""
LeRobot v3.0 Dataset Recorder & Multi-Modal Exporter for VLA Training.
---------------------------------------------------------------------
Wraps the official LeRobotDataset API (LeRobot v0.4+) to automatically 
handle chunked AV1 video encoding, parquet metadata tables, and seamless 
Hugging Face Hub syncing.
"""

import os
from pathlib import Path
from typing import Optional
import numpy as np
from PIL import Image

from lerobot.datasets.lerobot_dataset import LeRobotDataset

class LeRobotDatasetRecorder:
    """
    Records teleoperated robot trajectories formatted natively for LeRobot v3.0 (v0.4.x).
    """

    def __init__(
        self,
        dataset_dir: str = "data/lerobot_dataset",
        fps: int = 30,
        task_description: str = "pick up the red cube and place it into the bin",
        image_height: int = 240,
        image_width: int = 320,
        repo_id: str = "local/dataset"
    ):
        self.dataset_dir = Path(dataset_dir)
        self.fps = fps
        self.default_task = task_description
        self.repo_id = repo_id
        self.image_height = image_height
        self.image_width = image_width

        self.features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": (3, self.image_height, self.image_width),
                "names": ["channels", "height", "width"]
            },
            "observation.images.extrinsic": {
                "dtype": "video",
                "shape": (3, self.image_height, self.image_width),
                "names": ["channels", "height", "width"]
            },
            "observation.images.topdown": {
                "dtype": "video",
                "shape": (3, self.image_height, self.image_width),
                "names": ["channels", "height", "width"]
            }
        }

        if self.dataset_dir.exists() and (self.dataset_dir / "meta" / "info.json").exists():
            self.dataset = LeRobotDataset(self.repo_id, root=self.dataset_dir)
        else:
            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.dataset_dir,
                features=self.features, vcodec="h264",
            )

        self.is_recording = False
        self.current_task = self.default_task

    def start_recording(self, task_description: Optional[str] = None) -> None:
        """Start recording a new episode."""
        self.current_task = task_description or self.default_task
        self.is_recording = True
        print(f"[LeRobot Dataset] Started recording episode (Task: '{self.current_task}')")

    def record_step(
        self,
        state: np.ndarray,
        action: np.ndarray,
        wrist_rgb: np.ndarray,
        wrist_depth: np.ndarray,
        extrinsic_rgb: np.ndarray,
        topdown_rgb: np.ndarray,
    ) -> None:
        """Record a single synchronized timestep with all modalities."""
        if not self.is_recording:
            return

        frame_dict = {
            "observation.state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "observation.images.wrist": Image.fromarray(wrist_rgb),
            "observation.images.extrinsic": Image.fromarray(extrinsic_rgb),
            "observation.images.topdown": Image.fromarray(topdown_rgb),
            "task": self.current_task
        }

        # Note: wrist_depth is intentionally omitted here as native LeRobot v0.4.x 
        # compresses all images to AV1 video, which destroys floating-point metric depth.
        # If depth is required for future architectures, it must be saved as a separate tensor.
        
        self.dataset.add_frame(frame_dict)

    def save_episode(self, success: bool = True) -> Optional[str]:
        """Commit the episode to disk using LeRobotDataset chunked video encoding."""
        if not self.is_recording:
            return None

        self.dataset.save_episode()
        self.is_recording = False
        print("[LeRobot Dataset] Episode saved and chunked to disk natively.")
        return str(self.dataset_dir)

    def discard_episode(self) -> None:
        """Discard the current episode."""
        if not self.is_recording:
            return
        self.dataset.clear_episode_buffer()
        self.is_recording = False
        print("[LeRobot Dataset] Episode discarded.")
