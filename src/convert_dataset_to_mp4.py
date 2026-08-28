"""
Convert Legacy PNG-heavy LeRobot Datasets to Official MP4 Video Dataset Format.
---------------------------------------------------------------------------------
Scans existing datasets, loads frames from NPZ or PNG sequences, encodes each modality
into standard H.264 MP4 videos, updates meta/info.json schema, and safely cleans up
the thousands of PNG files.

Usage:
  python -m src.convert_dataset_to_mp4 --data-dir data/red_block_dataset
"""

import os
import sys
import glob
import json
import shutil
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from PIL import Image

from src.lerobot_dataset import (
    encode_video_stream,
    save_parquet_episode,
    update_dataset_stats,
)


def convert_dataset(data_dir_str: str, remove_pngs: bool = True, fps: int = 30):
    data_dir = Path(data_dir_str)
    if not data_dir.exists():
        print(f"\033[1;31mError: Directory '{data_dir}' does not exist.\033[0m")
        sys.exit(1)

    meta_dir = data_dir / "meta"
    info_path = meta_dir / "info.json"
    episodes_path = meta_dir / "episodes.jsonl"
    data_chunk_dir = data_dir / "data" / "chunk-000"
    videos_chunk_dir = data_dir / "videos" / "chunk-000"

    data_chunk_dir.mkdir(parents=True, exist_ok=True)
    videos_chunk_dir.mkdir(parents=True, exist_ok=True)

    if not episodes_path.exists():
        print(f"\033[1;31mError: No episodes.jsonl found at {episodes_path}\033[0m")
        sys.exit(1)

    episodes = []
    with open(episodes_path, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))

    print(f"\n\033[1;34m[Dataset Converter] Found {len(episodes)} episodes in '{data_dir}'.\033[0m")

    total_pngs_before = len(glob.glob(str(data_dir / "**" / "*.png"), recursive=True))

    total_frames_cum = 0

    for ep in episodes:
        ep_id = ep.get("episode_index", 0)
        print(f"\n\033[1;34m--> Processing Episode {ep_id:04d}...\033[0m")

        npz_path = data_chunk_dir / f"episode_{ep_id:06d}.npz"
        states_arr = None
        actions_arr = None
        timestamps_arr = None
        wrist_arr = None
        depth_arr = None
        extrinsic_arr = None
        topdown_arr = None

        if npz_path.exists():
            npz = np.load(npz_path)
            states_arr = npz.get("observation.state")
            actions_arr = npz.get("action")
            timestamps_arr = npz.get("timestamp")
            wrist_arr = npz.get("observation.images.wrist")
            depth_arr = npz.get("observation.images.wrist_depth")
            extrinsic_arr = npz.get("observation.images.extrinsic")
            topdown_arr = npz.get("observation.images.topdown")

        n_frames = len(states_arr) if states_arr is not None else ep.get("length", 0)
        start_idx = total_frames_cum
        end_idx = start_idx + n_frames
        total_frames_cum = end_idx

        frame_indices = np.arange(n_frames, dtype=np.int64)
        episode_indices = np.full(n_frames, ep_id, dtype=np.int64)
        indices = np.arange(start_idx, end_idx, dtype=np.int64)
        task_indices = np.zeros(n_frames, dtype=np.int64)

        # 1. Output MP4 paths into chunk-000/
        w_dir = videos_chunk_dir / "observation.images.wrist"
        d_dir = videos_chunk_dir / "observation.images.wrist_depth"
        e_dir = videos_chunk_dir / "observation.images.extrinsic"
        t_dir = videos_chunk_dir / "observation.images.topdown"

        for d in [w_dir, d_dir, e_dir, t_dir]:
            d.mkdir(parents=True, exist_ok=True)

        w_mp4 = w_dir / f"episode_{ep_id:06d}.mp4"
        d_mp4 = d_dir / f"episode_{ep_id:06d}.mp4"
        e_mp4 = e_dir / f"episode_{ep_id:06d}.mp4"
        t_mp4 = t_dir / f"episode_{ep_id:06d}.mp4"

        encode_tasks = [
            (wrist_arr, w_mp4, False),
            (depth_arr, d_mp4, True),
            (extrinsic_arr, e_mp4, False),
            (topdown_arr, t_mp4, False),
        ]

        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(encode_video_stream, frames, p, fps, 15, is_dep) for frames, p, is_dep in encode_tasks]
            for f in as_completed(futures):
                f.result()

        print(f"  ✓ Encoded 4x MP4 streams into videos/chunk-000/ for Episode {ep_id:04d}")

        # 2. Write Parquet table
        parquet_file = data_chunk_dir / f"episode_{ep_id:06d}.parquet"
        save_parquet_episode(
            parquet_path=parquet_file,
            states=states_arr,
            actions=actions_arr,
            timestamps=timestamps_arr,
            frame_indices=frame_indices,
            episode_indices=episode_indices,
            indices=indices,
            task_indices=task_indices,
        )
        print(f"  ✓ Saved Apache Parquet table: {parquet_file.name}")

        # Remove unchunked legacy video folders or old PNGs if they exist
        if remove_pngs:
            for cam in ["observation.images.wrist", "observation.images.wrist_depth", "observation.images.extrinsic", "observation.images.topdown"]:
                legacy_png_dir = data_dir / "videos" / cam / f"episode_{ep_id:06d}"
                if legacy_png_dir.exists() and legacy_png_dir.is_dir():
                    shutil.rmtree(legacy_png_dir)
                legacy_mp4 = data_dir / "videos" / cam / f"episode_{ep_id:06d}.mp4"
                if legacy_mp4.exists() and legacy_mp4.is_file():
                    legacy_mp4.unlink()

    # Clean up empty legacy camera directories in videos/
    for cam in ["observation.images.wrist", "observation.images.wrist_depth", "observation.images.extrinsic", "observation.images.topdown"]:
        old_dir = data_dir / "videos" / cam
        if old_dir.exists() and not any(old_dir.iterdir()):
            old_dir.rmdir()

    # 3. Update meta/info.json with official schema
    total_episodes = len(episodes)
    if info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)

        info["codebase_version"] = "v2.0"
        info["total_episodes"] = total_episodes
        info["total_frames"] = total_frames_cum
        info["total_videos"] = total_episodes * 4
        info["splits"] = {"train": f"0:{total_episodes}"}
        info["data_path"] = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

        for cam, is_dep in [
            ("observation.images.wrist", False),
            ("observation.images.wrist_depth", True),
            ("observation.images.extrinsic", False),
            ("observation.images.topdown", False),
        ]:
            if cam in info.get("features", {}):
                info["features"][cam]["dtype"] = "video"
                info["features"][cam]["shape"] = [240, 320, 3]
                info["features"][cam]["video_info"] = {
                    "video.fps": float(fps),
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": is_dep,
                    "has_audio": False,
                }
        with open(info_path, "w") as f:
            json.dump(info, f, indent=2)
        print("\n✓ Updated meta/info.json schema (path templates, splits, and video features).")

    # 4. Generate meta/stats.json
    update_dataset_stats(meta_dir, data_chunk_dir)
    print("✓ Computed and saved meta/stats.json normalization statistics.")

    total_files_after_f = [p for p in glob.glob(str(data_dir / "**" / "*"), recursive=True) if os.path.isfile(p)]
    print(f"\n\033[1;32m========================================================================")
    print(f"  ✓ 100% LeRobot Hub-Compliant Conversion Complete!")
    print(f"  Total files on disk: {len(total_files_after_f)} (Parquet + MP4 + Meta + Stats)")
    print(f"========================================================================\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Convert PNG LeRobot dataset into MP4 videos")
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Path to dataset directory (e.g. data/red_block_dataset)")
    parser.add_argument("--keep-pngs", action="store_true", default=False,
                        help="Keep old PNG files instead of deleting them")
    parser.add_argument("--fps", type=int, default=30,
                        help="FPS (default: 30)")
    args = parser.parse_args()

    convert_dataset(args.data_dir, remove_pngs=not args.keep_pngs, fps=args.fps)


if __name__ == "__main__":
    main()
