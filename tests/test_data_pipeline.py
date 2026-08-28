"""Unit tests for LeRobot data pipeline.

Tests cover:
- HDF5 dataset creation and loading
- Trajectory recording and saving
- Dataset statistics and filtering
- Train/val/test splits
- Metadata tracking
"""

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.lerobot_dataset import LeRobotHDF5Dataset, TrajectoryMetadata, create_dataset
from data.recorder import TrajectoryRecorder
from data.utils import (
    compute_dataset_statistics,
    create_train_val_test_split,
    filter_trajectories,
    load_trajectory,
    sample_trajectories,
    validate_trajectory_format,
)


class TestTrajectoryMetadata(unittest.TestCase):
    """Test TrajectoryMetadata dataclass."""

    def test_metadata_creation(self):
        """Test creating metadata."""
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp="2024-08-24T12:00:00",
            success=True,
            notes="Test episode",
        )

        self.assertEqual(metadata.task_id, "reaching")
        self.assertEqual(metadata.episode_id, 0)
        self.assertTrue(metadata.success)

    def test_metadata_to_dict(self):
        """Test metadata serialization."""
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp="2024-08-24T12:00:00",
            success=True,
        )

        data_dict = metadata.to_dict()
        self.assertIn("task_id", data_dict)
        self.assertEqual(data_dict["task_id"], "reaching")

    def test_metadata_from_dict(self):
        """Test metadata deserialization."""
        data_dict = {
            "task_id": "reaching",
            "episode_id": 0,
            "timestamp": "2024-08-24T12:00:00",
            "success": True,
            "notes": "Test",
            "date": "2024-08-24",
        }

        metadata = TrajectoryMetadata.from_dict(data_dict)
        self.assertEqual(metadata.task_id, "reaching")
        self.assertTrue(metadata.success)


class TestLeRobotHDF5Dataset(unittest.TestCase):
    """Test HDF5 dataset functionality."""

    def setUp(self):
        """Create temporary file for testing."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = Path(self.temp_dir.name) / "test_dataset.h5"

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()

    def test_create_and_open_dataset(self):
        """Test creating and opening a dataset."""
        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            self.assertIsNotNone(dataset._file)

        # Verify file exists
        self.assertTrue(self.filepath.exists())

    def test_add_trajectory(self):
        """Test adding a trajectory."""
        obs_data = np.random.randn(10, 8).astype(np.float32)
        act_data = np.random.randn(10, 6).astype(np.float32)
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp=datetime.now().isoformat(),
            success=True,
        )

        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            dataset.add_trajectory(
                {"state": obs_data},
                act_data,
                metadata,
            )

        self.assertTrue(self.filepath.exists())

    def test_load_trajectory(self):
        """Test loading a trajectory."""
        obs_data = np.random.randn(10, 8).astype(np.float32)
        act_data = np.random.randn(10, 6).astype(np.float32)
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp=datetime.now().isoformat(),
            success=True,
            notes="Test trajectory",
        )

        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            dataset.add_trajectory(
                {"state": obs_data},
                act_data,
                metadata,
            )

        # Load trajectory
        with LeRobotHDF5Dataset(str(self.filepath), mode="r") as dataset:
            loaded_obs, loaded_act, loaded_meta = dataset.load_trajectory(0)

            np.testing.assert_array_almost_equal(loaded_obs["state"], obs_data)
            np.testing.assert_array_almost_equal(loaded_act, act_data)
            self.assertEqual(loaded_meta.task_id, "reaching")
            self.assertTrue(loaded_meta.success)

    def test_multiple_observation_types(self):
        """Test storing multiple observation types."""
        obs_state = np.random.randn(10, 8).astype(np.float32)
        obs_position = np.random.randn(10, 3).astype(np.float32)
        act_data = np.random.randn(10, 6).astype(np.float32)
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp=datetime.now().isoformat(),
            success=True,
        )

        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            dataset.add_trajectory(
                {"state": obs_state, "position": obs_position},
                act_data,
                metadata,
            )

        with LeRobotHDF5Dataset(str(self.filepath), mode="r") as dataset:
            loaded_obs, _, _ = dataset.load_trajectory(0)

            self.assertIn("state", loaded_obs)
            self.assertIn("position", loaded_obs)

    def test_get_trajectory_ids(self):
        """Test getting all trajectory IDs."""
        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            for i in range(3):
                obs_data = np.random.randn(5, 8).astype(np.float32)
                act_data = np.random.randn(5, 6).astype(np.float32)
                metadata = TrajectoryMetadata(
                    task_id="reaching",
                    episode_id=i,
                    timestamp=datetime.now().isoformat(),
                    success=True,
                )

                dataset.add_trajectory(
                    {"state": obs_data},
                    act_data,
                    metadata,
                )

        with LeRobotHDF5Dataset(str(self.filepath), mode="r") as dataset:
            ids = dataset.get_trajectory_ids()
            self.assertEqual(len(ids), 3)

    def test_get_num_trajectories(self):
        """Test getting trajectory count."""
        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            for i in range(5):
                obs_data = np.random.randn(5, 8).astype(np.float32)
                act_data = np.random.randn(5, 6).astype(np.float32)
                metadata = TrajectoryMetadata(
                    task_id="reaching",
                    episode_id=i,
                    timestamp=datetime.now().isoformat(),
                    success=i % 2 == 0,
                )

                dataset.add_trajectory(
                    {"state": obs_data},
                    act_data,
                    metadata,
                )

        with LeRobotHDF5Dataset(str(self.filepath), mode="r") as dataset:
            count = dataset.get_num_trajectories()
            self.assertEqual(count, 5)

    def test_mismatched_lengths(self):
        """Test error handling for mismatched observation/action lengths."""
        obs_data = np.random.randn(10, 8).astype(np.float32)
        act_data = np.random.randn(5, 6).astype(np.float32)
        metadata = TrajectoryMetadata(
            task_id="reaching",
            episode_id=0,
            timestamp=datetime.now().isoformat(),
            success=True,
        )

        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            with self.assertRaises(ValueError):
                dataset.add_trajectory(
                    {"state": obs_data},
                    act_data,
                    metadata,
                )


class TestTrajectoryRecorder(unittest.TestCase):
    """Test trajectory recording."""

    def setUp(self):
        """Create recorder instance."""
        self.recorder = TrajectoryRecorder(
            task_id="reaching",
            observation_keys=["state", "position"],
            action_dim=6,
        )

    def test_record_single_episode(self):
        """Test recording a single episode."""
        self.recorder.start_episode()

        for i in range(10):
            obs = {
                "state": np.random.randn(8),
                "position": np.random.randn(3),
            }
            act = np.random.randn(6)
            self.recorder.record(obs, act)

        episode_id = self.recorder.end_episode(success=True)

        self.assertEqual(episode_id, 0)
        self.assertEqual(self.recorder.get_num_episodes(), 1)
        self.assertEqual(self.recorder.get_num_timesteps(), 10)

    def test_record_multiple_episodes(self):
        """Test recording multiple episodes."""
        for ep in range(3):
            self.recorder.start_episode()
            for i in range(5 + ep):  # Varying lengths
                obs = {
                    "state": np.random.randn(8),
                    "position": np.random.randn(3),
                }
                act = np.random.randn(6)
                self.recorder.record(obs, act)

            self.recorder.end_episode(success=ep % 2 == 0)

        self.assertEqual(self.recorder.get_num_episodes(), 3)
        self.assertEqual(self.recorder.get_num_timesteps(), 5 + 6 + 7)

    def test_record_without_active_episode(self):
        """Test error when recording without active episode."""
        obs = {
            "state": np.random.randn(8),
            "position": np.random.randn(3),
        }
        act = np.random.randn(6)

        with self.assertRaises(RuntimeError):
            self.recorder.record(obs, act)

    def test_get_episode(self):
        """Test retrieving episode data."""
        self.recorder.start_episode()

        obs_list = []
        act_list = []
        for i in range(10):
            obs = {
                "state": np.random.randn(8),
                "position": np.random.randn(3),
            }
            act = np.random.randn(6)
            obs_list.append(obs)
            act_list.append(act)
            self.recorder.record(obs, act)

        self.recorder.end_episode(success=True)

        episode_data = self.recorder.get_episode(0)

        self.assertIn("observations", episode_data)
        self.assertIn("actions", episode_data)
        self.assertIn("metadata", episode_data)
        self.assertEqual(episode_data["observations"]["state"].shape[0], 10)

    def test_get_statistics(self):
        """Test getting recorder statistics."""
        for ep in range(3):
            self.recorder.start_episode()
            for i in range(10):
                obs = {
                    "state": np.random.randn(8),
                    "position": np.random.randn(3),
                }
                act = np.random.randn(6)
                self.recorder.record(obs, act)

            self.recorder.end_episode(success=ep < 2)

        stats = self.recorder.get_statistics()

        self.assertEqual(stats["num_episodes"], 3)
        self.assertEqual(stats["num_timesteps"], 30)
        self.assertAlmostEqual(stats["success_rate"], 2/3, places=5)

    def test_save_to_hdf5(self):
        """Test saving to HDF5."""
        temp_dir = tempfile.TemporaryDirectory()
        filepath = Path(temp_dir.name) / "test_recording.h5"

        self.recorder.start_episode()
        for i in range(10):
            obs = {
                "state": np.random.randn(8),
                "position": np.random.randn(3),
            }
            act = np.random.randn(6)
            self.recorder.record(obs, act)

        self.recorder.end_episode(success=True)

        saved_path = self.recorder.save_to_hdf5(str(filepath))

        self.assertTrue(Path(saved_path).exists())
        temp_dir.cleanup()

    def test_action_size_validation(self):
        """Test action size validation."""
        self.recorder.start_episode()

        obs = {
            "state": np.random.randn(8),
            "position": np.random.randn(3),
        }
        act = np.random.randn(5)  # Wrong size

        with self.assertRaises(ValueError):
            self.recorder.record(obs, act)


class TestDatasetUtils(unittest.TestCase):
    """Test dataset utility functions."""

    def setUp(self):
        """Create test dataset."""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.filepath = Path(self.temp_dir.name) / "test_dataset.h5"

        # Create dataset with multiple trajectories
        with LeRobotHDF5Dataset(str(self.filepath), mode="w-") as dataset:
            for i in range(10):
                obs_data = np.random.randn(10 + i, 8).astype(np.float32)
                act_data = np.random.randn(10 + i, 6).astype(np.float32)
                success = i < 7  # 7 successful, 3 failed

                metadata = TrajectoryMetadata(
                    task_id="reaching" if i < 5 else "grasping",
                    episode_id=i,
                    timestamp=datetime.now().isoformat(),
                    success=success,
                )

                dataset.add_trajectory(
                    {"state": obs_data},
                    act_data,
                    metadata,
                )

    def tearDown(self):
        """Clean up."""
        self.temp_dir.cleanup()

    def test_load_trajectory(self):
        """Test loading single trajectory."""
        obs, act, meta = load_trajectory(str(self.filepath), 0)

        self.assertIn("state", obs)
        self.assertEqual(act.shape[1], 6)
        self.assertEqual(meta.task_id, "reaching")

    def test_compute_statistics(self):
        """Test computing dataset statistics."""
        stats = compute_dataset_statistics(str(self.filepath))

        self.assertEqual(stats["num_trajectories"], 10)
        self.assertGreater(stats["num_timesteps"], 0)
        self.assertAlmostEqual(stats["success_rate"], 0.7, places=5)

    def test_compute_statistics_success_filter(self):
        """Test statistics with success filter."""
        stats = compute_dataset_statistics(str(self.filepath), filter_success=True)

        self.assertEqual(stats["num_trajectories"], 7)

    def test_filter_by_task_id(self):
        """Test filtering by task ID."""
        filtered = filter_trajectories(str(self.filepath), task_id="reaching")

        self.assertEqual(len(filtered), 5)

    def test_filter_by_success(self):
        """Test filtering by success."""
        filtered = filter_trajectories(str(self.filepath), success_only=True)

        self.assertEqual(len(filtered), 7)

    def test_filter_combined(self):
        """Test combined filtering."""
        filtered = filter_trajectories(
            str(self.filepath),
            task_id="reaching",
            success_only=True,
        )

        self.assertEqual(len(filtered), 5)

    def test_create_train_val_test_split(self):
        """Test creating splits."""
        split = create_train_val_test_split(
            str(self.filepath),
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
            seed=42,
        )

        self.assertIn("train", split)
        self.assertIn("val", split)
        self.assertIn("test", split)

        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        self.assertEqual(total, 10)

    def test_split_determinism(self):
        """Test that splits are deterministic with same seed."""
        split1 = create_train_val_test_split(str(self.filepath), seed=42)
        split2 = create_train_val_test_split(str(self.filepath), seed=42)

        self.assertEqual(split1["train"], split2["train"])
        self.assertEqual(split1["val"], split2["val"])
        self.assertEqual(split1["test"], split2["test"])

    def test_split_different_with_different_seed(self):
        """Test that different seeds produce different splits."""
        split1 = create_train_val_test_split(str(self.filepath), seed=42)
        split2 = create_train_val_test_split(str(self.filepath), seed=43)

        self.assertNotEqual(split1["train"], split2["train"])

    def test_sample_trajectories(self):
        """Test sampling trajectories."""
        sampled = sample_trajectories(str(self.filepath), n_samples=5, seed=42)

        self.assertEqual(len(sampled), 5)

    def test_sample_determinism(self):
        """Test that sampling is deterministic."""
        sample1 = sample_trajectories(str(self.filepath), n_samples=5, seed=42)
        sample2 = sample_trajectories(str(self.filepath), n_samples=5, seed=42)

        self.assertEqual(sample1, sample2)

    def test_validate_trajectory(self):
        """Test trajectory validation."""
        is_valid = validate_trajectory_format(
            str(self.filepath),
            0,
            required_obs_keys=["state"],
        )

        self.assertTrue(is_valid)


class TestEndToEnd(unittest.TestCase):
    """End-to-end integration tests."""

    def test_record_and_load_workflow(self):
        """Test full workflow: record -> save -> load."""
        temp_dir = tempfile.TemporaryDirectory()
        filepath = Path(temp_dir.name) / "workflow_test.h5"

        # Record
        recorder = TrajectoryRecorder(
            task_id="reaching",
            observation_keys=["state"],
            action_dim=6,
        )

        for ep in range(3):
            recorder.start_episode()
            for i in range(10):
                obs = {"state": np.random.randn(8)}
                act = np.random.randn(6)
                recorder.record(obs, act)

            recorder.end_episode(success=True)

        recorder.save_to_hdf5(str(filepath))

        # Load and verify
        with LeRobotHDF5Dataset(str(filepath), mode="r") as dataset:
            ids = dataset.get_trajectory_ids()
            self.assertEqual(len(ids), 3)

            for ep_id in ids:
                obs, act, meta = dataset.load_trajectory(ep_id)
                self.assertEqual(act.shape[0], 10)
                self.assertEqual(meta.task_id, "reaching")

        temp_dir.cleanup()

    def test_large_dataset_handling(self):
        """Test handling of larger dataset."""
        temp_dir = tempfile.TemporaryDirectory()
        filepath = Path(temp_dir.name) / "large_dataset.h5"

        with LeRobotHDF5Dataset(str(filepath), mode="w-") as dataset:
            for i in range(50):
                obs_data = np.random.randn(100, 20).astype(np.float32)
                act_data = np.random.randn(100, 6).astype(np.float32)

                metadata = TrajectoryMetadata(
                    task_id=f"task_{i % 5}",
                    episode_id=i,
                    timestamp=datetime.now().isoformat(),
                    success=i % 3 != 0,
                )

                dataset.add_trajectory(
                    {"state": obs_data},
                    act_data,
                    metadata,
                )

        # Verify stats
        stats = compute_dataset_statistics(str(filepath))
        self.assertEqual(stats["num_trajectories"], 50)

        # Test splits
        split = create_train_val_test_split(str(filepath), seed=42)
        total = len(split["train"]) + len(split["val"]) + len(split["test"])
        self.assertEqual(total, 50)

        temp_dir.cleanup()


if __name__ == "__main__":
    unittest.main()
