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


def resolve_checkpoint_path(path_str: str) -> Path:
    """Resolve checkpoint directory, automatically appending pretrained_model if needed."""
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
    video_dir: str = "eval_videos"
):
    ckpt_dir = resolve_checkpoint_path(checkpoint_path)
    if not (ckpt_dir / "config.json").exists():
        print(f"\033[1;31mError: Could not find config.json in '{ckpt_dir}'.\033[0m")
        return

    print(f"\n\033[1;34m========================================================================\033[0m")
    print(f"\033[1;34m         Agilex Piper Interactive LeRobot Policy Evaluator             \033[0m")
    print(f"\033[1;34m========================================================================\033[0m")
    print(f"  Checkpoint Path : {ckpt_dir}")
    print(f"  Target Episodes : {num_episodes}")
    print(f"  Max Steps/Ep    : {max_steps}")
    print(f"  Headless Mode   : {headless}")
    print(f"  Save Videos     : {save_video}")
    print(f"\033[1;34m========================================================================\033[0m\n")

    # Load policy and pre/post processors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Policy Evaluator] Loading pretrained policy model onto {device}...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    policy_cfg.device = str(device)
    
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    policy_cls = get_policy_class(policy_cfg.type)
    policy = policy_cls.from_pretrained(str(ckpt_dir))
    policy.eval()
    policy.to(device)

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg,
        pretrained_path=str(ckpt_dir),
        preprocessor_overrides={"device_processor": {"device": str(device)}},
        postprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    print("✓ Policy model and pre/post processors successfully loaded!")

    # Initialize environment
    from src.environment.env import PiperGymEnv
    render_mode = "rgb_array" if headless else "human"
    env = PiperGymEnv(render_mode=render_mode)

    viewer = None
    if not headless and VIEWER_AVAILABLE:
        print("[Visualizer] Opening interactive 3D GLFW MuJoCo viewer window...")
        viewer = mujoco.viewer.launch_passive(env.env.model, env.env.data)

    success_count = 0
    video_out_dir = Path(video_dir)
    if save_video:
        video_out_dir.mkdir(parents=True, exist_ok=True)

    task_description = getattr(env, "task_description", "pick up the red block and place in blue bin")

    for ep in range(num_episodes):
        print(f"\n--- Episode {ep+1}/{num_episodes} ---")
        res = env.reset()
        obs_raw = res[0] if isinstance(res, tuple) else res
        policy.reset()

        frames = []
        episode_reward = 0.0
        success = False

        for step_idx in range(max_steps):
            # Render visual frame for viewer or video recording
            if viewer is not None and viewer.is_running():
                viewer.sync()
                time.sleep(0.02)
            
            if save_video:
                rgb_frame = env.render()
                if rgb_frame is not None:
                    frames.append(rgb_frame)

            # Helper tensor formatters
            def _fmt_state(arr):
                if arr is None:
                    return None
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
                if t.dtype == torch.uint8:
                    t = t.float() / 255.0
                return t

            # Format raw observation dict for preprocessor
            if isinstance(obs_raw, dict):
                st = obs_raw.get("observation.state", obs_raw.get("agent_pos", obs_raw.get("joint_positions", None)))
                wr = obs_raw.get("observation.images.wrist", obs_raw.get("wrist", obs_raw.get("pixels", {}).get("wrist", None)))
                ex = obs_raw.get("observation.images.extrinsic", obs_raw.get("extrinsic", obs_raw.get("pixels", {}).get("extrinsic", None)))

                obs_formatted = {
                    "observation.state": _fmt_state(st),
                    "task": task_description,
                }
                if wr is not None:
                    obs_formatted["observation.images.wrist"] = _fmt_img(wr)
                if ex is not None:
                    obs_formatted["observation.images.extrinsic"] = _fmt_img(ex)
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
            write_video(v_path, frames, fps=30)
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
    args = parser.parse_args()

    evaluate_policy(
        args.checkpoint,
        num_episodes=args.num_episodes,
        max_steps=args.max_steps,
        headless=args.headless,
        save_video=args.save_video,
        video_dir=args.video_dir
    )


if __name__ == "__main__":
    main()
