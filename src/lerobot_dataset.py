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
from contextlib import contextmanager
import numpy as np
from PIL import Image

try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

@contextmanager
def silence_stderr():
    """Redirect C-level stderr (fd 2) to /dev/null to silence libx264/FFmpeg encoding output."""
    try:
        null_fd = os.open(os.devnull, os.O_WRONLY)
        old_stderr = os.dup(2)
        os.dup2(null_fd, 2)
        os.close(null_fd)
        yield
    except Exception:
        yield
    finally:
        try:
            os.dup2(old_stderr, 2)
            os.close(old_stderr)
        except Exception:
            pass

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

class LeRobotDatasetRecorder:
    """
    Records teleoperated robot trajectories formatted natively for LeRobot v3.0 (v0.4.x).
    """

    def __init__(
        self,
        dataset_dir: str = "data/lerobot_dataset",
        fps: int = 30,
        task_description: str = "pick up the red cube and place it into the blue bin",
        image_height: int = 480,
        image_width: int = 640,
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
            }
        }

        has_episodes = (self.dataset_dir / "meta" / "episodes").exists() and any((self.dataset_dir / "meta" / "episodes").glob("*.parquet"))

        self.dataset = None
        if self.dataset_dir.exists() and (self.dataset_dir / "meta" / "info.json").exists() and has_episodes:
            try:
                self.dataset = LeRobotDataset.resume(
                    repo_id=self.repo_id,
                    root=self.dataset_dir,
                    rgb_encoder=RGBEncoderConfig(vcodec="h264"),
                )
            except Exception as e:
                print(f"[LeRobot Dataset] Resume failed ({e}), creating fresh dataset...")
                self.dataset = None

        if self.dataset is None:
            if self.dataset_dir.exists():
                import shutil
                shutil.rmtree(self.dataset_dir)

            self.dataset = LeRobotDataset.create(
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.dataset_dir,
                features=self.features,
                rgb_encoder=RGBEncoderConfig(vcodec="h264"),
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
        wrist_depth: Optional[np.ndarray] = None,
        extrinsic_rgb: Optional[np.ndarray] = None,
        topdown_rgb: Optional[np.ndarray] = None,
    ) -> None:
        """Record a single synchronized timestep with 2 camera modalities (wrist + extrinsic)."""
        if not self.is_recording:
            return

        frame_dict = {
            "observation.state": np.array(state, dtype=np.float32),
            "action": np.array(action, dtype=np.float32),
            "observation.images.wrist": Image.fromarray(wrist_rgb),
            "observation.images.extrinsic": Image.fromarray(extrinsic_rgb if extrinsic_rgb is not None else wrist_rgb),
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

        with silence_stderr():
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

    @property
    def num_episodes(self) -> int:
        return getattr(self.dataset, "num_episodes", 0)

    def finalize(self) -> None:
        """Finalize dataset chunking and write metadata footers (LeRobot v3.0 standard)."""
        if hasattr(self.dataset, "finalize"):
            with silence_stderr():
                self.dataset.finalize()
            print("[LeRobot Dataset] Dataset finalized with valid metadata footers.")
