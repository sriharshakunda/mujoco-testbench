import argparse
import gc
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def migrate():
    parser = argparse.ArgumentParser(description="Migrate LeRobot v2.0 dataset to v3.0 chunked schema")
    parser.add_argument("--repo-id", type=str, required=True, help="Hugging Face Dataset Repo ID (e.g. username/dataset_name)")
    parser.add_argument("--v2-root", type=str, default="data/red_block_dataset", help="Input v2.0 dataset path")
    parser.add_argument("--v3-root", type=str, default="data/dataset_v3", help="Output v3.0 dataset path")
    parser.add_argument("--vcodec", type=str, default="h264", help="Video codec (default: h264)")
    args = parser.parse_args()

    v2_root = Path(args.v2_root)
    v3_root = Path(args.v3_root)

    with open(v2_root / "meta" / "info.json") as f:
        info = json.load(f)

    features = info["features"]
    fps = info.get("fps", 30)

    # Remove auto-generated keys from feature spec
    for k in ["task_index", "episode_index", "frame_index", "timestamp", "index"]:
        features.pop(k, None)

    for k, v in features.items():
        if "shape" in v and isinstance(v["shape"], list):
            v["shape"] = tuple(v["shape"])

    ds = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=fps,
        root=v3_root,
        features=features,
        vcodec=args.vcodec,
    )

    episodes_df = pd.read_parquet(v2_root / "meta" / "episodes.parquet")
    tasks_df = pd.read_parquet(v2_root / "meta" / "tasks.parquet")
    tasks_dict = dict(zip(tasks_df["task_index"], tasks_df["task"]))

    for _, ep_row in episodes_df.iterrows():
        ep_idx = int(ep_row["episode_index"])
        t = ep_row["tasks"][0] if isinstance(ep_row["tasks"], (list, np.ndarray)) else ep_row["tasks"]

        try:
            task_idx = int(t)
            task_str = tasks_dict[task_idx]
        except ValueError:
            task_str = str(t)

        ep_parquet = v2_root / "data" / "chunk-000" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(ep_parquet)

        import av
        videos = {}
        for key in features:
            if "images" in key:
                vid_path = v2_root / "videos" / "chunk-000" / key / f"episode_{ep_idx:06d}.mp4"
                if vid_path.exists():
                    container = av.open(str(vid_path))
                    stream = container.streams.video[0]
                    frames = []
                    for frame in container.decode(stream):
                        frames.append(frame.to_ndarray(format="rgb24"))
                    videos[key] = frames
                    container.close()

        print(f"Loaded episode {ep_idx}, {len(df)} frames")
        for i in range(len(df)):
            frame_dict = {}
            for key in features:
                if "images" in key:
                    frame_dict[key] = Image.fromarray(videos[key][i])
                else:
                    val = df.iloc[i][key]
                    if isinstance(val, (list, np.ndarray)):
                        frame_dict[key] = np.array(val, dtype=np.float32)
                    else:
                        frame_dict[key] = val

            frame_dict["task"] = task_str
            ds.add_frame(frame_dict)

        ds.save_episode()
        print(f"Saved episode {ep_idx} to v3!")
        gc.collect()

    print("✓ Migration complete! Dataset saved to", v3_root)


if __name__ == "__main__":
    migrate()
