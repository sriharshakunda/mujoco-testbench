"""
Ultra-Fast Offline 4-Modality & Telemetry Graph Dataset Visualizer for LeRobot VLA Pipeline.
-------------------------------------------------------------------------------------------
Synchronously replays recorded episodes with all 4 visual modalities + synchronized telemetry chart:
  - Row 1: [1. Gripper View 2D RGB (Wrist)]       | [2. Gripper View 3D Metric Depth (Plasma)]
  - Row 2: [3. Side View 2D RGB (Extrinsic)]      | [4. Top-Down Overview 2D RGB (Table Coverage)]
  - Row 3: [5. Synchronized 7-DOF Telemetry Graph (q1..q6 + Gripper mm) with Interactive Playhead]

Interactive Controls:
  [Space]       : Play / Pause Replay
  [Left/Right]  : Step 1 frame backward / forward (Frame Scrubbing)
  [Up/Down]     : Previous / Next episode (Instant <10ms Switch)
  [R]           : Restart current episode from frame 0
  [Q / ESC]     : Exit visualizer
"""

import os
import sys
import glob
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Dict, Any, List
import numpy as np
from PIL import Image, ImageDraw

from src.telemetry_plotter import TelemetryGraphPlotter


def load_dataset_metadata(data_dir: Path):
    """Load info.json and episodes.jsonl metadata."""
    info_path = data_dir / "meta" / "info.json"
    episodes_path = data_dir / "meta" / "episodes.jsonl"

    if not episodes_path.exists():
        print(f"\033[1;31mError: No episodes found at {episodes_path}\033[0m")
        return None, []

    with open(info_path, "r") as f:
        info = json.load(f)

    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    return info, episodes


class OfflineDatasetVisualizer:
    """Interactive offline visualizer replay engine for LeRobot VLA recorded episodes."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.info, self.episodes = load_dataset_metadata(self.data_dir)
        if not self.episodes:
            raise ValueError(f"No recorded episodes found in {self.data_dir}")

        self.cur_ep_idx = 0
        self.cur_frame_idx = 0
        self.is_playing = True
        self.fps = self.info.get("fps", 30)

        self.plotter = TelemetryGraphPlotter(width=640, height=240)

        # Load first episode
        self._load_episode(0)

        # Initialize visualizer window (640x720)
        from src.camera_visualizer import MultiCameraVisualizer
        self.viz = MultiCameraVisualizer(
            title="LeRobot VLA Dataset Visualizer [4-Modality + Live Graph Replay]",
            include_graph=True,
        )

    def _load_episode(self, ep_idx: int):
        """Load video frames, 3D depth, and states for an episode with instant NPZ & parallel fallback."""
        self.cur_ep_idx = max(0, min(ep_idx, len(self.episodes) - 1))
        self.cur_frame_idx = 0
        ep_info = self.episodes[self.cur_ep_idx]
        ep_id = ep_info.get("episode_index", self.cur_ep_idx)

        t0 = time.time()
        sys.stdout.write(f"\r\033[K\033[1;34m[Visualizer] Loading Episode {ep_id:04d}/{len(self.episodes)-1:04d} ({ep_info.get('length', 0)} frames)...\033[0m")
        sys.stdout.flush()

        npz_path = self.data_dir / "data" / "chunk-000" / f"episode_{ep_id:06d}.npz"
        self.states = None
        self.actions = None
        loaded_from_npz = False

        # 1. Fast path: Instant load from NPZ archive (<10ms)
        if npz_path.exists():
            try:
                npz_data = np.load(npz_path)
                self.states = npz_data.get("observation.state", None)
                self.actions = npz_data.get("action", None)

                if "observation.images.wrist" in npz_data:
                    self.wrist_frames = npz_data["observation.images.wrist"]
                    self.depth_frames = npz_data["observation.images.wrist_depth"]
                    self.extrinsic_frames = npz_data["observation.images.extrinsic"]
                    self.topdown_frames = npz_data["observation.images.topdown"]
                    self.n_frames = len(self.wrist_frames)
                    loaded_from_npz = True
            except Exception:
                pass

        # 2. Parallel fallback: Load PNG sequences using ThreadPoolExecutor
        if not loaded_from_npz:
            wrist_pngs = sorted(glob.glob(str(self.data_dir / "videos" / "observation.images.wrist" / f"episode_{ep_id:06d}" / "*.png")))
            depth_pngs = sorted(glob.glob(str(self.data_dir / "videos" / "observation.images.wrist_depth" / f"episode_{ep_id:06d}" / "*.png")))
            extrinsic_pngs = sorted(glob.glob(str(self.data_dir / "videos" / "observation.images.extrinsic" / f"episode_{ep_id:06d}" / "*.png")))
            topdown_pngs = sorted(glob.glob(str(self.data_dir / "videos" / "observation.images.topdown" / f"episode_{ep_id:06d}" / "*.png")))

            def _read_img(p):
                return np.array(Image.open(p))

            with ThreadPoolExecutor(max_workers=8) as executor:
                if wrist_pngs:
                    self.wrist_frames = list(executor.map(_read_img, wrist_pngs))
                    self.n_frames = len(self.wrist_frames)
                else:
                    self.wrist_frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * 10
                    self.n_frames = 10

                if depth_pngs:
                    depth_raw = list(executor.map(_read_img, depth_pngs))
                    self.depth_frames = [img.astype(np.float32) / 1000.0 for img in depth_raw]
                else:
                    self.depth_frames = [np.ones((240, 320), dtype=np.float32) * 0.5] * self.n_frames

                if extrinsic_pngs:
                    self.extrinsic_frames = list(executor.map(_read_img, extrinsic_pngs))
                else:
                    self.extrinsic_frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * self.n_frames

                if topdown_pngs:
                    self.topdown_frames = list(executor.map(_read_img, topdown_pngs))
                else:
                    self.topdown_frames = [np.zeros((240, 320, 3), dtype=np.uint8)] * self.n_frames

        elapsed = time.time() - t0
        sys.stdout.write(f"\r\033[K\033[1;32m✓ Episode {ep_id:04d} Ready ({self.n_frames} frames loaded in {elapsed*1000:.1f}ms)\033[0m\n")
        sys.stdout.flush()

    def _overlay_telemetry_banner(self, img_arr: np.ndarray, f_idx: int) -> np.ndarray:
        """Draw clean telemetry banner on top of the top-down frame."""
        img = Image.fromarray(img_arr)
        draw = ImageDraw.Draw(img)

        # Bottom telemetry bar
        draw.rectangle([(0, 208), (320, 240)], fill=(0, 0, 0))

        ep_info = self.episodes[self.cur_ep_idx]
        status = "PLAY" if self.is_playing else "PAUSE"
        col = (0, 255, 120) if self.is_playing else (255, 100, 100)

        draw.text((8, 214), f"Ep {self.cur_ep_idx:02d} | Frm {f_idx:03d}/{self.n_frames-1:03d}", fill=(255, 255, 255))
        draw.text((190, 214), f"[{status}]", fill=col)

        if self.states is not None and f_idx < len(self.states):
            grip_mm = self.states[f_idx][6] * 1000.0
            draw.text((245, 214), f"G:{grip_mm:3.0f}m", fill=(0, 220, 255))

        return np.array(img)

    def run(self):
        """Main visualization loop."""
        import glfw

        print("\n" + "=" * 68)
        print("    LeRobot VLA Dataset 4-Modality & Telemetry Graph Replay Viewer")
        print("=" * 68)
        print("  - Row 1: Gripper View 2D RGB (Wrist)  | Gripper View 3D Depth (Colormap)")
        print("  - Row 2: Side View Extrinsic 2D RGB    | Top-Down Table Coverage RGB")
        print("  - Row 3: Live 7-DOF Telemetry Chart (q1..q6 + Gripper mm) with Playhead")
        print("--------------------------------------------------------------------")
        print("  [Space]       — Play / Pause")
        print("  [Left/Right]  — Step 1 frame backward / forward (Scrub)")
        print("  [Up/Down]     — Previous / Next episode (Instant Switch)")
        print("  [R]           — Restart current episode from frame 0")
        print("  [Q / ESC]     — Quit visualizer")
        print("=" * 68 + "\n")

        last_time = time.time()
        frame_interval = 1.0 / self.fps

        while self.viz.is_open:
            now = time.time()

            # Advance playback if playing
            if self.is_playing and (now - last_time >= frame_interval):
                last_time = now
                self.cur_frame_idx = (self.cur_frame_idx + 1) % max(1, self.n_frames)

            # Render current 4 modalities + live graph
            f_idx = min(self.cur_frame_idx, self.n_frames - 1)
            wrist_rgb = self.wrist_frames[f_idx]
            wrist_dep = self.depth_frames[f_idx]
            extrinsic_rgb = self.extrinsic_frames[f_idx]
            topdown_rgb = self._overlay_telemetry_banner(self.topdown_frames[f_idx], f_idx)

            # Render graph with playhead on frame f_idx
            graph_rgb = self.plotter.render_replay_graph(self.states, f_idx)

            # Update visualizer dashboard (640x720)
            self.viz.update(wrist_rgb, wrist_dep, extrinsic_rgb, topdown_rgb, graph_rgb=graph_rgb)

            # Check GLFW keyboard events
            if self.viz.window is not None:
                w = self.viz.window
                if glfw.get_key(w, glfw.KEY_ESCAPE) == glfw.PRESS or glfw.get_key(w, glfw.KEY_Q) == glfw.PRESS:
                    break
                if glfw.get_key(w, glfw.KEY_SPACE) == glfw.PRESS:
                    self.is_playing = not self.is_playing
                    time.sleep(0.15)
                if glfw.get_key(w, glfw.KEY_RIGHT) == glfw.PRESS:
                    self.is_playing = False
                    self.cur_frame_idx = min(self.cur_frame_idx + 1, self.n_frames - 1)
                    time.sleep(0.08)
                if glfw.get_key(w, glfw.KEY_LEFT) == glfw.PRESS:
                    self.is_playing = False
                    self.cur_frame_idx = max(self.cur_frame_idx - 1, 0)
                    time.sleep(0.08)
                if glfw.get_key(w, glfw.KEY_DOWN) == glfw.PRESS:
                    self._load_episode(self.cur_ep_idx + 1)
                    time.sleep(0.2)
                if glfw.get_key(w, glfw.KEY_UP) == glfw.PRESS:
                    self._load_episode(self.cur_ep_idx - 1)
                    time.sleep(0.2)
                if glfw.get_key(w, glfw.KEY_R) == glfw.PRESS:
                    self.cur_frame_idx = 0
                    time.sleep(0.15)

            time.sleep(0.005)

        self.viz.close()
        print("\nVisualizer closed.")


def main():
    parser = argparse.ArgumentParser(description="Offline 4-Modality LeRobot VLA Dataset Visualizer")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_dataset",
                        help="Path to recorded dataset directory (default: data/lerobot_dataset)")
    args = parser.parse_args()

    viz = OfflineDatasetVisualizer(data_dir=args.data_dir)
    viz.run()


if __name__ == "__main__":
    main()
