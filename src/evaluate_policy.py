"""
Official Hugging Face LeRobot Policy Evaluation Benchmark for Agilex Piper Arm.
--------------------------------------------------------------------------------
Evaluates trained LeRobot policies (ACT, Diffusion Policy) in closed-loop MuJoCo
simulation across randomized test episodes.

Features:
  - Loads official LeRobot pretrained models (`ACTPolicy.from_pretrained`) or Hub repositories
  - Multi-camera visual rendering (Wrist RGB, Depth, Scene, Topdown)
  - Closed-loop action chunking execution via `policy.select_action(observation)`
  - Automatic goal verification (cube placed inside blue target bin)
  - Rollout video saving and live visualization

Usage:
  ./docker_run.sh --eval --checkpoint checkpoints/act_lerobot/best_model --num-episodes 10
  ./docker_run.sh --eval --checkpoint checkpoints/act_lerobot/best_model --headless --save-video
"""

import os
import sys
import time
import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import torch
import mujoco

from src.environment.env import PiperEnv, N_ARM_JOINTS, N_GRIPPER
from src.camera import WristCamera

from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception:
    VIEWER_AVAILABLE = False


TARGET_BIN_POS = np.array([0.35, 0.32, 0.15])


def evaluate_lerobot_policy(
    checkpoint_path: str = "checkpoints/act_lerobot/best_model",
    num_episodes: int = 10,
    max_steps: int = 350,
    headless: bool = False,
    save_video: bool = False,
    video_dir: str = "eval_videos",
    device: Optional[str] = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("\n" + "=" * 76)
    print("      Hugging Face LeRobot Closed-Loop Policy Evaluation Benchmark")
    print("=" * 76)
    print(f"  Checkpoint Path   : \033[1;34m{checkpoint_path}\033[0m")
    print(f"  Evaluation Device : \033[1;32m{device.upper()}\033[0m")
    print(f"  Test Episodes     : \033[1;36m{num_episodes}\033[0m")
    print(f"  Max Steps/Episode : \033[1;36m{max_steps} frames\033[0m")
    print(f"  Headless Mode     : \033[1;33m{headless}\033[0m")
    print("=" * 76 + "\n")

    # 1. Load Pretrained LeRobot Policy & Normalization Processors
    try:
        policy = ACTPolicy.from_pretrained(checkpoint_path)
    except Exception:
        try:
            policy = DiffusionPolicy.from_pretrained(checkpoint_path)
        except Exception:
            try:
                policy = SmolVLAPolicy.from_pretrained(checkpoint_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load LeRobot policy from {checkpoint_path}: {e}")

    policy.to(device)
    policy.eval()
    print(f"✓ Loaded LeRobot {policy.__class__.__name__} from {checkpoint_path}\n")

    # Load preprocessor & postprocessor (handles normalization / unnormalization)
    from lerobot.policies.factory import make_pre_post_processors
    try:
        preprocessor, postprocessor = make_pre_post_processors(policy.config, pretrained_path=checkpoint_path)
        print("✓ Loaded policy preprocessor & postprocessor from checkpoint.")
    except Exception as e:
        print(f"Note: Checkpoint did not contain preprocessors ({e}). Loading fallback stats from data/red_block_dataset...")
        import json
        fallback_stats_path = Path("data/red_block_dataset/meta/stats.json")
        if not fallback_stats_path.exists():
            fallback_stats_path = Path("data/lerobot_dataset/meta/stats.json")
        
        dataset_stats = None
        if fallback_stats_path.exists():
            with open(fallback_stats_path, "r") as f:
                dataset_stats = json.load(f)
        
        preprocessor, postprocessor = make_pre_post_processors(
            policy.config,
            dataset_stats=dataset_stats,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "normalizer_processor": {"norm_map": policy.config.normalization_mapping},
            },
            postprocessor_overrides={
                "unnormalizer_processor": {"norm_map": policy.config.normalization_mapping},
            },
        )
        print("✓ Configured fallback preprocessor & postprocessor pipelines.")

    # 2. Setup Environment & Cameras
    env = PiperEnv()
    cube_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")

    wrist_cam = WristCamera(env.model, "wrist_rgb", exposure=1.0)
    scene_cam = WristCamera(env.model, "scene_cam", exposure=1.0)
    topdown_cam = WristCamera(env.model, "topdown_cam", exposure=1.0)

    viewer = None
    if not headless and VIEWER_AVAILABLE:
        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -22
        except Exception as e:
            print(f"Note: Running headless (viewer error: {e})")
            viewer = None

    if save_video:
        import imageio
        v_path = Path(video_dir)
        v_path.mkdir(parents=True, exist_ok=True)

    success_count = 0
    substeps = 5  # 5 * 2ms = 10ms physics step per 30fps frame

    try:
        for ep in range(1, num_episodes + 1):
            print(f"\n\033[1;34m--- [Episode {ep:02d}/{num_episodes:02d}] -------------------------------------------\033[0m")
            env.reset(randomize_cubes=True)
            # Reset policy internal state buffer (action queues, CVAE sampling)
            policy.reset()

            # Set arm ready posture
            env.data.qpos[:6] = np.array([0.20, -2.00, -0.60, 0.00, 1.00, 0.00])
            env.data.qvel[:] = 0.0
            env.data.ctrl[:6] = env.data.qpos[:6]
            env.data.ctrl[6] = 0.04
            mujoco.mj_forward(env.model, env.data)

            step_idx = 0
            ep_frames = []
            ep_success = False

            ctrl_min = env.model.actuator_ctrlrange[:, 0]
            ctrl_max = env.model.actuator_ctrlrange[:, 1]

            while step_idx < max_steps:
                # 1. Capture observation state & images
                raw_state = np.concatenate([env.data.qpos[:N_ARM_JOINTS], [env.data.qpos[N_ARM_JOINTS]]])
                w_rgb, w_dep = wrist_cam.get_rgb_and_depth(env.data)
                s_rgb = scene_cam.get_rgb(env.data)
                t_rgb = topdown_cam.get_rgb(env.data)

                if save_video:
                    ep_frames.append(s_rgb)

                # Format observations expected by LeRobot policy
                # Note: LeRobot processors expect raw unbatched observations (1D tensors / (C, H, W) images)
                obs = {
                    "observation.state": torch.from_numpy(raw_state.astype(np.float32)),
                    "observation.images.wrist": torch.from_numpy(w_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0),
                    "observation.images.wrist_depth": torch.from_numpy(w_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0),
                    "observation.images.extrinsic": torch.from_numpy(s_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0),
                    "observation.images.topdown": torch.from_numpy(t_rgb.astype(np.float32).transpose(2, 0, 1) / 255.0),
                    "task": "pick up the red cube and place it into the bin",
                }

                # Adapt camera keys for SmolVLA if needed
                if hasattr(policy.config, "image_features"):
                    policy_cams = list(policy.config.image_features.keys())
                    if "observation.images.camera1" in policy_cams:
                        obs["observation.images.camera1"] = obs["observation.images.wrist"]
                        obs["observation.images.camera2"] = obs["observation.images.extrinsic"]
                        obs["observation.images.camera3"] = obs["observation.images.topdown"]

                # Process observation through normalizer & device placement
                processed_obs = preprocessor(obs)

                # 2. Select next action from LeRobot policy
                with torch.no_grad():
                    action_tensor = policy.select_action(processed_obs)  # (1, 7)
                    unnormalized_action = postprocessor(action_tensor)
                    action_cmd = unnormalized_action.squeeze(0).cpu().numpy()

                # 3. Apply action to MuJoCo robot
                env.data.ctrl[:7] = np.clip(action_cmd[:7], ctrl_min, ctrl_max)
                for _ in range(substeps):
                    mujoco.mj_step(env.model, env.data)

                if viewer is not None and step_idx % 2 == 0:
                    viewer.sync()

                step_idx += 1

                # 4. Check success condition (cube placed inside blue target bin)
                cube_pos = env.data.xpos[cube_id]
                dx = abs(cube_pos[0] - TARGET_BIN_POS[0])
                dy = abs(cube_pos[1] - TARGET_BIN_POS[1])
                dz = abs(cube_pos[2] - TARGET_BIN_POS[2])

                if dx < 0.08 and dy < 0.08 and dz < 0.08:
                    ep_success = True
                    # Hold for ~2 seconds so the viewer shows the cube settled in the bin
                    for _ in range(60):
                        mujoco.mj_step(env.model, env.data)
                        if viewer is not None:
                            viewer.sync()
                        if save_video:
                            ep_frames.append(scene_cam.get_rgb(env.data))
                    break

            if ep_success:
                success_count += 1
                final_cube = env.data.xpos[cube_id]
                print(f"  \033[1;32m✓ Episode {ep:02d} SUCCESS in {step_idx} steps! (Cube at [{final_cube[0]:.3f}, {final_cube[1]:.3f}, {final_cube[2]:.3f}])\033[0m")
            else:
                print(f"  \033[1;31m✗ Episode {ep:02d} FAILED (Timed out at {step_idx} steps)\033[0m")

            if save_video and ep_frames:
                out_vid = v_path / f"eval_episode_{ep:04d}.mp4"
                imageio.mimsave(str(out_vid), ep_frames, fps=30)
                print(f"  ✓ Saved rollout video: {out_vid}")

    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    success_rate = (success_count / num_episodes) * 100.0
    print("\n" + "=" * 76)
    print("  EVALUATION SUMMARY:")
    print(f"  Total Episodes : {num_episodes}")
    print(f"  Success Count  : {success_count}/{num_episodes}")
    print(f"  Success Rate   : \033[1;32m{success_rate:.1f}%\033[0m")
    print("=" * 76 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Trained LeRobot Policy in Piper MuJoCo Environment")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/act_lerobot/best_model",
                        help="Path to trained LeRobot policy checkpoint directory or HF repo")
    parser.add_argument("--num-episodes", type=int, default=10,
                        help="Number of evaluation test episodes (default: 10)")
    parser.add_argument("--max-steps", type=int, default=350,
                        help="Maximum execution steps per episode (default: 350)")
    parser.add_argument("--headless", action="store_true", default=False,
                        help="Run in headless mode without GUI viewer")
    parser.add_argument("--save-video", action="store_true", default=False,
                        help="Save evaluation rollout videos")
    parser.add_argument("--video-dir", type=str, default="eval_videos",
                        help="Directory to save evaluation rollout videos")
    args = parser.parse_args()

    evaluate_lerobot_policy(
        checkpoint_path=args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        save_video=args.save_video,
        video_dir=args.video_dir,
    )


if __name__ == "__main__":
    main()
