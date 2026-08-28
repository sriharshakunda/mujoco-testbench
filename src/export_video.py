"""
Export LeRobot VLA Dataset Episodes as Social-Media-Ready High-Quality MP4 Videos.
----------------------------------------------------------------------------------
Renders the complete 5-channel visualizer layout into a polished H.264 MP4 video:
  - Row 1: Wrist 2D RGB (320x240)        | Wrist 3D Metric Depth (320x240, Colormap)
  - Row 2: Side Extrinsic 2D RGB (320x240) | Top-Down Overview 2D RGB (320x240) + Task Banner
  - Row 3: Synchronized 7-DOF Telemetry Graph (640x240) with Moving Playhead

Supports:
  - Any episode index (--episode 0, --episode 1, etc.) or all episodes (--all)
  - Direct 1080p/2K upscaling for crisp social media rendering (--upscale 2)
  - Native ffmpeg pipe with H.264 yuv420p encoding (compatible with Twitter/X, YouTube, Premiere, Mac/Windows/iOS)
  - Animated tqdm progress bar
"""

import os
import sys
import glob
import json
import time
import argparse
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image, ImageDraw

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable=None, total=None, desc="", unit="", file=None, **kwargs):
        class SimpleProgress:
            def __init__(self, tot, description):
                self.total = tot or 1
                self.desc = description
                self.n = 0
            def __enter__(self):
                return self
            def __exit__(self, *args):
                sys.stdout.write("\n")
                sys.stdout.flush()
            def update(self, inc=1):
                self.n += inc
                pct = int(self.n / self.total * 100)
                sys.stdout.write(f"\r\033[K{self.desc}: {pct}% [{self.n}/{self.total} {unit}]")
                sys.stdout.flush()
        if iterable is not None:
            return iterable
        return SimpleProgress(total, desc)

from src.telemetry_plotter import TelemetryGraphPlotter
from src.camera_visualizer import depth_to_colormap


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


def load_episode_data(data_dir: Path, ep_id: int):
    """Load video frames, depth, and states for an episode."""
    npz_path = data_dir / "data" / "chunk-000" / f"episode_{ep_id:06d}.npz"
    states = None
    actions = None

    if npz_path.exists():
        try:
            npz_data = np.load(npz_path)
            states = npz_data.get("observation.state", None)
            actions = npz_data.get("action", None)
            wrist_frames = npz_data["observation.images.wrist"]
            depth_frames = npz_data["observation.images.wrist_depth"]
            extrinsic_frames = npz_data["observation.images.extrinsic"]
            topdown_frames = npz_data["observation.images.topdown"]
            return wrist_frames, depth_frames, extrinsic_frames, topdown_frames, states, actions
        except Exception:
            pass

    # Fallback to PNG sequences
    wrist_pngs = sorted(glob.glob(str(data_dir / "videos" / "observation.images.wrist" / f"episode_{ep_id:06d}" / "*.png")))
    depth_pngs = sorted(glob.glob(str(data_dir / "videos" / "observation.images.wrist_depth" / f"episode_{ep_id:06d}" / "*.png")))
    extrinsic_pngs = sorted(glob.glob(str(data_dir / "videos" / "observation.images.extrinsic" / f"episode_{ep_id:06d}" / "*.png")))
    topdown_pngs = sorted(glob.glob(str(data_dir / "videos" / "observation.images.topdown" / f"episode_{ep_id:06d}" / "*.png")))

    def _read_img(p):
        return np.array(Image.open(p))

    with ThreadPoolExecutor(max_workers=8) as executor:
        wrist_frames = list(executor.map(_read_img, wrist_pngs)) if wrist_pngs else []
        extrinsic_frames = list(executor.map(_read_img, extrinsic_pngs)) if extrinsic_pngs else []
        topdown_frames = list(executor.map(_read_img, topdown_pngs)) if topdown_pngs else []
        if depth_pngs:
            depth_raw = list(executor.map(_read_img, depth_pngs))
            depth_frames = [img.astype(np.float32) / 1000.0 for img in depth_raw]
        else:
            depth_frames = [np.ones((240, 320), dtype=np.float32) * 0.5] * len(wrist_frames)

    return wrist_frames, depth_frames, extrinsic_frames, topdown_frames, states, actions


def compose_frame(
    w_rgb: np.ndarray,
    w_dep: np.ndarray,
    s_rgb: np.ndarray,
    t_rgb: np.ndarray,
    graph_rgb: np.ndarray,
    ep_idx: int,
    frame_idx: int,
    total_frames: int,
    task_desc: str,
    fps: int = 30,
) -> np.ndarray:
    """Compose the 640x720 5-channel visualizer grid with labels."""
    frame_h, frame_w = 240, 320
    grid_h, grid_w = 720, 640

    # Depth colormap
    if len(w_dep.shape) == 2:
        depth_rgb = depth_to_colormap(w_dep)
    else:
        depth_rgb = w_dep

    grid = np.empty((grid_h, grid_w, 3), dtype=np.uint8)
    # Row 1: Wrist RGB & Wrist Depth
    grid[0:frame_h, 0:frame_w] = w_rgb
    grid[0:frame_h, frame_w:grid_w] = depth_rgb
    # Row 2: Scene Extrinsic RGB & Top-Down RGB
    grid[frame_h:frame_h*2, 0:frame_w] = s_rgb
    grid[frame_h:frame_h*2, frame_w:grid_w] = t_rgb
    # Row 3: Synchronized Telemetry Graph
    grid[frame_h*2:grid_h, 0:grid_w] = graph_rgb

    # Draw Overlays
    img = Image.fromarray(grid)
    draw = ImageDraw.Draw(img)

    # Sub-view labels
    labels = [
        (10, 8, "1. Gripper View 2D RGB"),
        (frame_w + 10, 8, "2. Gripper View 3D Metric Depth"),
        (10, frame_h + 8, "3. Side View 2D RGB (Extrinsic)"),
        (frame_w + 10, frame_h + 8, "4. Top-Down Overview 2D RGB"),
    ]
    for x, y, text in labels:
        draw.rectangle([(x - 4, y - 2), (x + len(text) * 7 + 4, y + 14)], fill=(0, 0, 0, 180))
        draw.text((x, y), text, fill=(255, 255, 255))

    # Top-Down Task Banner Overlay
    draw.rectangle([(frame_w, frame_h*2 - 26), (grid_w, frame_h*2)], fill=(0, 0, 0, 210))
    time_str = f"{(frame_idx/fps):.1f}s / {(total_frames/fps):.1f}s"
    draw.text((frame_w + 8, frame_h*2 - 20), f"Ep {ep_idx:02d} | Frm {frame_idx:03d}/{total_frames-1:03d} ({time_str})", fill=(0, 229, 255))

    return np.array(img)


def get_ffmpeg_binary() -> str:
    """Find available ffmpeg binary from imageio_ffmpeg, system PATH, or common locations."""
    try:
        import imageio_ffmpeg
        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            return exe
    except Exception:
        pass

    import shutil
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return sys_ffmpeg

    for p in ["/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg"]:
        if os.path.exists(p):
            return p

    raise FileNotFoundError("Could not find ffmpeg binary or imageio-ffmpeg package.")


def export_episode_to_video(
    data_dir: Path,
    ep_info: Dict[str, Any],
    output_path: Path,
    fps: int = 30,
    upscale: int = 1,
    crf: int = 18,
):
    """Render and encode an episode to an MP4 video using ffmpeg."""
    ep_id = ep_info.get("episode_index", 0)
    task_desc = ep_info.get("tasks", [""])[0]

    wrist_f, depth_f, extrinsic_f, topdown_f, states, actions = load_episode_data(data_dir, ep_id)
    n_frames = len(wrist_f)
    if n_frames == 0:
        print(f"\033[1;31mError: No frames found for Episode {ep_id}\033[0m")
        return

    plotter = TelemetryGraphPlotter(width=640, height=240)
    out_w, out_h = 640 * upscale, 720 * upscale

    output_path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_bin = get_ffmpeg_binary()

    # Launch ffmpeg sub-process with H.264 yuv420p encoding
    cmd = [
        ffmpeg_bin,
        "-y",                       # Overwrite output
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{out_w}x{out_h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "-",                  # Pipe stdin
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",      # Maximum compatibility (iOS, Mac, Windows, Web, Twitter/X)
        "-preset", "medium",
        "-crf", str(crf),           # High visual quality
        str(output_path),
    ]

    process = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)

    print(f"\n\033[1;34m[Export Video] Encoding Episode {ep_id:04d} ({n_frames} frames -> {output_path.name})...\033[0m")

    try:
        with tqdm(total=n_frames, desc=f"Rendering Video Ep {ep_id:04d}", unit="frames", file=sys.stdout) as pbar:
            for i in range(n_frames):
                graph_rgb = plotter.render_replay_graph(states, i)
                composed = compose_frame(
                    w_rgb=wrist_f[i],
                    w_dep=depth_f[i],
                    s_rgb=extrinsic_f[i],
                    t_rgb=topdown_f[i],
                    graph_rgb=graph_rgb,
                    ep_idx=ep_id,
                    frame_idx=i,
                    total_frames=n_frames,
                    task_desc=task_desc,
                    fps=fps,
                )

                if upscale > 1:
                    composed_img = Image.fromarray(composed).resize((out_w, out_h), Image.Resampling.LANCZOS)
                    frame_bytes = composed_img.tobytes()
                else:
                    frame_bytes = composed.tobytes()

                process.stdin.write(frame_bytes)
                pbar.update(1)

        process.stdin.close()
        process.wait()

        if process.returncode == 0:
            file_size_mb = output_path.stat().st_size / (1024 * 1024)
            print(f"\033[1;32m✓ Successfully Exported Video: {output_path} ({file_size_mb:.2f} MB, {n_frames/fps:.1f}s @ {out_w}x{out_h})\033[0m\n")
        else:
            _, stderr = process.communicate()
            print(f"\033[1;31mFFmpeg error: {stderr.decode()}\033[0m")

    except Exception as e:
        process.kill()
        raise e


def main():
    parser = argparse.ArgumentParser(description="Export LeRobot VLA Dataset Episodes as High-Quality MP4 Videos")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_dataset",
                        help="Path to recorded dataset directory (default: data/lerobot_dataset)")
    parser.add_argument("--episode", type=int, default=0,
                        help="Episode index to export as video (default: 0)")
    parser.add_argument("--all", action="store_true", default=False,
                        help="Export all recorded episodes in the dataset")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output MP4 file path (default: exports/episode_XXXXXX.mp4)")
    parser.add_argument("--upscale", type=int, default=1, choices=[1, 2],
                        help="Resolution upscale multiplier (1 = 640x720, 2 = 1280x1440 HD for social media)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Video framerate (default: 30)")
    parser.add_argument("--crf", type=int, default=18,
                        help="H.264 video quality CRF (default: 18, lower is higher quality)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    info, episodes = load_dataset_metadata(data_dir)
    if not episodes:
        print(f"\033[1;31mNo episodes found in {data_dir}\033[0m")
        return

    fps = info.get("fps", args.fps) if info else args.fps

    if args.all:
        for ep_info in episodes:
            ep_id = ep_info.get("episode_index", 0)
            out_file = Path(args.output) if args.output else data_dir / "exports" / f"episode_{ep_id:06d}.mp4"
            export_episode_to_video(data_dir, ep_info, out_file, fps=fps, upscale=args.upscale, crf=args.crf)
    else:
        ep_id = args.episode
        matching = [ep for ep in episodes if ep.get("episode_index", 0) == ep_id]
        if not matching:
            print(f"\033[1;31mEpisode {ep_id} not found. Available episodes: 0 to {len(episodes)-1}\033[0m")
            return
        ep_info = matching[0]
        out_file = Path(args.output) if args.output else data_dir / "exports" / f"episode_{ep_id:06d}.mp4"
        export_episode_to_video(data_dir, ep_info, out_file, fps=fps, upscale=args.upscale, crf=args.crf)


if __name__ == "__main__":
    main()
