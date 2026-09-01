"""
Offline Dataset Camera Re-Renderer for MuJoCo Piper Arm Testbench.
-------------------------------------------------------------------
Re-renders camera video streams (true wrist_rgb, scene_cam, front_cam) for existing
LeRobot datasets by replaying the exact joint state trajectories (qpos) through MuJoCo.

Zero manual re-collection required! Upgrades any dataset in seconds.

Usage:
  python -m src.rerender_dataset --dataset data/redcube_picknplace_manual_v4 --output data/redcube_picknplace_manual_v4_fixed
"""

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import mujoco
from PIL import Image

from src.environment.env import PiperEnv
from src.camera import WristCamera
from src.lerobot_dataset import LeRobotDatasetRecorder


def rerender_dataset(
    input_dataset_dir: str = "data/redcube_picknplace_manual_v4",
    output_dataset_dir: str = "data/redcube_picknplace_manual_v4_fixed",
):
    input_path = Path(input_dataset_dir)
    output_path = Path(output_dataset_dir)

    if not input_path.exists():
        print(f"Error: Input dataset directory '{input_dataset_dir}' does not exist.")
        return

    print("=" * 75)
    print("      Offline Dataset Camera Re-Renderer & Repair Tool")
    print("=" * 75)
    print(f"  Input Dataset  : {input_dataset_dir}")
    print(f"  Output Dataset : {output_dataset_dir}")
    print("=" * 75 + "\n")

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        input_dataset = LeRobotDataset(root=input_path, repo_id="local/input_demo")
    except Exception as e:
        print(f"Error loading input dataset via LeRobot: {e}")
        return

    num_episodes = input_dataset.num_episodes
    total_frames = len(input_dataset)
    print(f"✓ Loaded Input Dataset: {total_frames} total frames across {num_episodes} episodes.\n")

    env = PiperEnv(render_mode=None)
    wrist_cam = WristCamera(env.model, "wrist_rgb", height=480, width=640)
    side_cam = WristCamera(env.model, "scene_cam", height=480, width=640)
    front_cam = WristCamera(env.model, "front_cam", height=480, width=640)

    recorder = LeRobotDatasetRecorder(
        dataset_dir=str(output_path),
        fps=30,
        task_description="pick up the red cube and place it into the blue bin",
        image_height=480,
        image_width=640,
        repo_id="local/rerendered_dataset",
    )

    states = input_dataset.hf_dataset["observation.state"]
    actions = input_dataset.hf_dataset["action"]
    ep_indices = input_dataset.hf_dataset["episode_index"]

    current_ep = -1

    print("[Re-Renderer] Replaying joint trajectories and capturing TRUE Wrist Camera frames...")

    for i in range(total_frames):
        ep_idx = int(ep_indices[i])
        state = np.array(states[i], dtype=np.float32)
        action = np.array(actions[i], dtype=np.float32)

        if ep_idx != current_ep:
            if current_ep != -1:
                recorder.save_episode()
                print(f"  ✓ Re-rendered Episode {current_ep+1}/{num_episodes}")

            current_ep = ep_idx
            env.reset(randomize_cubes=True)
            recorder.start_recording()

        # Apply exact joint state to MuJoCo simulation
        env.data.qpos[:7] = state[:7]
        env.data.ctrl[:7] = state[:7]
        mujoco.mj_forward(env.model, env.data)

        # Capture TRUE Wrist, Side, and Front camera frames
        w_rgb = wrist_cam.get_rgb(env.data)
        s_rgb = side_cam.get_rgb(env.data)
        f_rgb = front_cam.get_rgb(env.data)

        recorder.record_step(
            state=state,
            action=action,
            wrist_rgb=w_rgb,
            extrinsic_rgb=s_rgb,
            topdown_rgb=f_rgb,
        )

    if current_ep != -1:
        recorder.save_episode()
        print(f"  ✓ Re-rendered Episode {current_ep+1}/{num_episodes}")

    env.close()

    print("\n" + "=" * 75)
    print(f"✓ Re-Rendering Complete! {num_episodes} episodes updated with TRUE Wrist Camera frames.")
    print(f"  Saved to: '{output_dataset_dir}'")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Offline Dataset Camera Re-Renderer")
    parser.add_argument("--dataset", type=str, default="data/redcube_picknplace_manual_v4", help="Input dataset path")
    parser.add_argument("--output", type=str, default="data/redcube_picknplace_manual_v4_fixed", help="Output dataset path")
    args = parser.parse_args()

    rerender_dataset(input_dataset_dir=args.dataset, output_dataset_dir=args.output)


if __name__ == "__main__":
    main()
