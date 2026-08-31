"""
Interactive Policy Evaluator with Real-Time MuJoCo 3D Visualizer & Video Recording.
----------------------------------------------------------------------------------
Evaluates trained LeRobot policies (SmolVLA, Diffusion, ACT) in closed-loop MuJoCo simulation
with live 3D window visualization and optional MP4 video export.

Usage:
    # Live Interactive 3D Viewer:
    python -m src.evaluate_policy --checkpoint outputs/train/smolvla_piper/checkpoints/last/pretrained_model --num-episodes 5

    # Headless Mode with Video Recording:
    python -m src.evaluate_policy --checkpoint outputs/train/smolvla_piper/checkpoints/last/pretrained_model --num-episodes 5 --headless --save-video
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch

try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

import mujoco
try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception:
    VIEWER_AVAILABLE = False

from src.environment.env import PiperEnv
from lerobot.policies.factory import make_policy
from lerobot.configs.policies import PreTrainedConfig
from lerobot.utils.io_utils import write_video


def resolve_checkpoint_path(path_str: str):
    """Resolve checkpoint directory or HuggingFace repo ID."""
    if path_str.startswith("lerobot/") or "/" in path_str and not Path(path_str).exists():
        return path_str
    p = Path(path_str).resolve()
    if (p / "pretrained_model").exists():
        return p / "pretrained_model"
    if p.is_symlink() or p.is_dir():
        target = p.resolve()
        if (target / "pretrained_model").exists():
            return target / "pretrained_model"
    return p


def evaluate_policy(
    checkpoint_path: str,
    num_episodes: int = 5,
    max_steps: int = 300,
    headless: bool = False,
    save_video: bool = False,
    video_dir: str = "eval_videos",
    init_pose: str = "auto",
    wrist_cam_source: str = "front",
    extrinsic_cam_source: str = "scene",
    task: str = "pick up the red cube and place it into the blue bin",
    fps: int = 60,
):
    ckpt_target = resolve_checkpoint_path(checkpoint_path)
    ckpt_dir_str = str(ckpt_target)

    # Determine initial reset posture
    start_qpos = np.array([0.0, -3.14, -0.22, 0.0, 0.0, 0.0, 0.04])  # HOME default
    if init_pose == "forward":
        start_qpos = np.array([0.20, -2.00, -0.60, 0.00, 1.00, 0.00, 0.04])
    elif init_pose == "home":
        start_qpos = np.array([0.0, -3.14, -0.22, 0.0, 0.0, 0.0, 0.04])
    elif init_pose == "auto" and isinstance(ckpt_target, Path):
        train_cfg_path = ckpt_target / "train_config.json"
        if train_cfg_path.exists():
            try:
                import json
                with open(train_cfg_path) as f:
                    t_cfg = json.load(f)
                repo_id = str(t_cfg.get("dataset", {}).get("repo_id", "")).lower()
                root_path = str(t_cfg.get("dataset", {}).get("root", "")).lower()
                # Datasets recorded with forward posture
                if ("redcube_picknplace" in repo_id or "redcube_picknplace" in root_path) and "100ep" not in repo_id and "100ep" not in root_path and "manual" not in repo_id and "manual" not in root_path:
                    start_qpos = np.array([0.20, -2.00, -0.60, 0.00, 1.00, 0.00, 0.04])
            except Exception:
                pass

    print(f"\n\033[1;34m========================================================================\033[0m")
    print(f"\033[1;34m         Agilex Piper Interactive LeRobot Policy Evaluator             \033[0m")
    print(f"\033[1;34m========================================================================\033[0m")
    print(f"  Checkpoint Path : {ckpt_dir_str}")
    print(f"  Task Prompt     : {task}")
    print(f"  Target Episodes : {num_episodes}")
    print(f"  Max Steps/Ep    : {max_steps}")
    print(f"  Headless Mode   : {headless}")
    print(f"  Save Videos     : {save_video} (FPS: {fps})")
    print(f"  Init Pose Qpos  : {start_qpos[:6].round(2)}")
    print(f"  Wrist Cam Source: {wrist_cam_source}")
    print(f"  Extrinsic Source: {extrinsic_cam_source}")
    print(f"\033[1;34m========================================================================\033[0m\n")

    # Load policy and pre/post processors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Policy Evaluator] Loading pretrained policy model onto {device}...")
    policy_cfg = PreTrainedConfig.from_pretrained(ckpt_dir_str)
    policy_cfg.device = str(device)
    
    # Ensure state feature shape matches environment 7-dim state
    if "observation.state" in policy_cfg.input_features:
        policy_cfg.input_features["observation.state"].shape = (7,)

    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(ckpt_dir_str)
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=ckpt_dir_str,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("✓ Policy model and pre/post processors successfully loaded!")

    # Initialize environment
    from src.environment.env import PiperGymEnv
    render_mode = "rgb_array" if headless else "human"
    env = PiperGymEnv(
        render_mode=render_mode,
        wrist_cam_source=wrist_cam_source,
        extrinsic_cam_source=extrinsic_cam_source,
    )
    viewer = getattr(env.env, "_viewer", None)

    success_count = 0
    video_out_dir = Path(video_dir)
    if save_video:
        video_out_dir.mkdir(parents=True, exist_ok=True)

    task_description = task

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        res = env.reset()

        # Apply dataset starting posture
        env.env.data.qpos[:7] = start_qpos
        env.env.data.ctrl[:7] = env.env.data.qpos[:7]
        mujoco.mj_forward(env.env.model, env.env.data)

        obs_raw = env._get_obs(wrist_source=wrist_cam_source)
        policy.reset()

        frames = []
        episode_reward = 0.0
        success = False

        for step_idx in range(max_steps):
            if save_video:
                frame = env.render(camera_name="scene_cam")
                if frame is not None:
                    frames.append(frame)

            # Helper tensor formatters
            def _fmt_state(arr):
                if arr is None:
                    return None
                expected_state_dim = 7
                if "observation.state" in policy_cfg.input_features:
                    expected_state_dim = policy_cfg.input_features["observation.state"].shape[0]
                if len(arr) < expected_state_dim:
                    arr = np.pad(arr, (0, expected_state_dim - len(arr)))
                t = torch.as_tensor(arr, dtype=torch.float32, device=device)
                return t.unsqueeze(0) if t.ndim == 1 else t

            def _fmt_img(arr):
                if arr is None:
                    return None
                t = torch.as_tensor(arr, device=device)
                if t.ndim == 3:  # (H, W, C) -> (1, C, H, W)
                    t = t.permute(2, 0, 1).unsqueeze(0)
                elif t.ndim == 4 and t.shape[-1] == 3:  # (B, H, W, C) -> (B, C, H, W)
                    t = t.permute(0, 3, 1, 2)
                # Normalize RGB tensors to float32 [0.0, 1.0] range for PyTorch VLM vision backbones
                t = t.float()
                if t.max() > 1.0:
                    t = t / 255.0
                return t
            if isinstance(obs_raw, dict):
                st = obs_raw.get("observation.state", obs_raw.get("agent_pos", obs_raw.get("joint_positions", None)))
                wr = obs_raw.get("observation.images.wrist", obs_raw.get("wrist", obs_raw.get("pixels", {}).get("wrist", None)))
                ex = obs_raw.get("observation.images.extrinsic", obs_raw.get("extrinsic", obs_raw.get("pixels", {}).get("extrinsic", None)))
                tp = obs_raw.get("observation.images.topdown", obs_raw.get("topdown", obs_raw.get("pixels", {}).get("topdown", None)))

                obs_formatted = {
                    "observation.state": _fmt_state(st),
                    "task": task_description,
                }
                if wr is not None:
                    obs_formatted["observation.images.wrist"] = _fmt_img(wr)
                    obs_formatted["observation.images.camera1"] = _fmt_img(wr)
                    obs_formatted["observation.images.left_wrist_0_rgb"] = _fmt_img(wr)
                if ex is not None:
                    obs_formatted["observation.images.extrinsic"] = _fmt_img(ex)
                    obs_formatted["observation.images.camera2"] = _fmt_img(ex)
                    obs_formatted["observation.images.base_0_rgb"] = _fmt_img(ex)
                if tp is not None:
                    obs_formatted["observation.images.topdown"] = _fmt_img(tp)
                    obs_formatted["observation.images.camera3"] = _fmt_img(tp)
                    obs_formatted["observation.images.right_wrist_0_rgb"] = _fmt_img(tp)
            else:
                obs_formatted = {
                    "observation.state": _fmt_state(obs_raw),
                    "task": task_description,
                }

            with torch.no_grad():
                obs_proc = preprocessor(obs_formatted)
                action_tensor = policy.select_action(obs_proc)
                if postprocessor is not None:
                    action_tensor = postprocessor(action_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()

            # Handle action shape for Pi0 (32-dim) or SmolVLA (6-dim)
            if len(action) == 6:
                action = np.append(action, 0.04)
            elif len(action) > 7:
                action = action[:7]

            res_step = env.step(action)
            if isinstance(res_step, tuple) and len(res_step) == 5:
                obs_raw, reward, terminated, truncated, info = res_step
                done = terminated or truncated
            elif isinstance(res_step, tuple) and len(res_step) == 4:
                obs_raw, reward, done, info = res_step
            else:
                obs_raw, reward, done, info = res_step, 0.0, False, {}

            episode_reward += reward

            if isinstance(info, dict) and info.get("success", False):
                success = True
                print(f"  ✓ SUCCESS! Pick & Place completed at step {step_idx+1}")
                break

        if success:
            success_count += 1
        else:
            print(f"  Episode finished (Steps: {step_idx+1}, Total Reward: {episode_reward:.2f})")

        if save_video and len(frames) > 0:
            v_path = video_out_dir / f"eval_episode_{ep+1}.mp4"
            write_video(v_path, frames, fps=fps)
            print(f"  ✓ Saved rollout video: {v_path}")

    if viewer is not None:
        viewer.close()

    success_rate = (success_count / num_episodes) * 100.0
    print(f"\n========================================================================")
    print(f"               Evaluation Benchmark Summary                            ")
    print(f"========================================================================")
    print(f"  Total Episodes : {num_episodes}")
    print(f"  Successes      : {success_count} / {num_episodes}")
    print(f"  Success Rate   : {success_rate:.1f}%")
    print(f"========================================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Agilex Piper Interactive LeRobot Policy Evaluator")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint folder or pretrained_model")
    parser.add_argument("--num-episodes", type=int, default=5, help="Number of evaluation episodes")
    parser.add_argument("--max-steps", type=int, default=300, help="Maximum steps per episode")
    parser.add_argument("--headless", action="store_true", help="Run without opening interactive 3D viewer")
    parser.add_argument("--save-video", action="store_true", help="Export rollout MP4 videos")
    parser.add_argument("--video-dir", type=str, default="eval_videos", help="Directory to save MP4 videos")
    parser.add_argument("--fps", type=int, default=60, help="Playback FPS for exported rollout MP4 videos (default: 60)")
    parser.add_argument("--init-pose", type=str, default="auto", choices=["auto", "home", "forward"], help="Initial arm posture: 'auto' (detect from dataset), 'home', or 'forward'")
    parser.add_argument("--wrist-cam", type=str, default="front", choices=["front", "wrist", "scene", "topdown"], help="Camera source for 'observation.images.wrist' feature key")
    parser.add_argument("--extrinsic-cam", type=str, default="scene", choices=["scene", "front", "topdown", "wrist"], help="Camera source for 'observation.images.extrinsic' feature key")
    args = parser.parse_args()

    evaluate_policy(
        args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        save_video=args.save_video,
        video_dir=args.video_dir,
        init_pose=args.init_pose,
        wrist_cam_source=args.wrist_cam,
        extrinsic_cam_source=args.extrinsic_cam,
        fps=args.fps,
    )


if __name__ == "__main__":
    main()
