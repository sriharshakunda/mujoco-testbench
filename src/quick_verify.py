"""
Quick Policy Verification Tool (#1 Offline Action MSE + #2 Trajectory Trace).
-----------------------------------------------------------------------------
Evaluates policy checkpoints in under 5 seconds to catch un-converged runs,
camera misalignments, or bad action outputs before launching long rollouts.

Usage:
  python -m src.quick_verify --checkpoint outputs/train/act_piper/checkpoints/002500/pretrained_model
"""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
import mujoco

from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from src.environment.env import PiperGymEnv
from src.evaluate_policy import resolve_checkpoint_path


def verify_checkpoint(
    checkpoint_path: str,
    dataset_root: str = None,
    num_samples: int = 5,
    trace_steps: int = 30,
):
    ckpt_dir = resolve_checkpoint_path(checkpoint_path)
    if not (ckpt_dir / "config.json").exists():
        print(f"\033[1;31mError: Could not find config.json in '{ckpt_dir}'.\033[0m")
        return

    print("\n\033[1;34m========================================================================\033[0m")
    print("\033[1;34m            Agilex Piper Rapid Policy Verification Tool                \033[0m")
    print("\033[1;34m========================================================================\033[0m")
    print(f"  Checkpoint Path : {ckpt_dir}")

    # Inspect dataset root from train_config.json if not provided
    train_cfg_path = ckpt_dir / "train_config.json"
    dataset_repo = None
    if train_cfg_path.exists():
        try:
            with open(train_cfg_path) as f:
                t_cfg = json.load(f)
            if not dataset_root:
                dataset_root = t_cfg.get("dataset", {}).get("root")
            dataset_repo = t_cfg.get("dataset", {}).get("repo_id")
        except Exception:
            pass

    print(f"  Dataset Root    : {dataset_root or 'Not specified'}")
    print("\033[1;34m========================================================================\033[0m\n")

    # Load Model onto CPU/CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[QuickVerify] Loading policy model on {device}...")
    policy_cfg = PreTrainedConfig.from_pretrained(str(ckpt_dir))
    policy_cfg.device = str(device)

    # Force 7-dim state compatibility
    if "observation.state" in policy_cfg.input_features:
        policy_cfg.input_features["observation.state"].shape = (7,)

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
    print("✓ Model and preprocessor successfully loaded!\n")

    # =========================================================================
    # TEST #1: OFFLINE ACTION MSE CHECK ON DATASET SAMPLES
    # =========================================================================
    print("\033[1;33m[Test #1] Running Offline Action Prediction MSE Check...\033[0m")
    mse_score = None
    if dataset_root and Path(dataset_root).exists():
        try:
            dataset = LeRobotDataset(repo_id=dataset_repo or "local/dataset", root=dataset_root)
            total_frames = len(dataset)
            indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)

            mses = []
            for idx in indices:
                sample = dataset[int(idx)]
                gt_action = sample["action"].numpy() if isinstance(sample["action"], torch.Tensor) else np.array(sample["action"])

                # Prepare observation dict for preprocessor
                st = sample["observation.state"]
                wr = sample.get("observation.images.wrist")
                ex = sample.get("observation.images.extrinsic")

                t_st = torch.as_tensor(st, dtype=torch.float32, device=device)
                if t_st.ndim == 1:
                    t_st = t_st.unsqueeze(0)

                obs_formatted = {"observation.state": t_st, "task": "pick up the red cube"}
                if wr is not None:
                    t_wr = torch.as_tensor(wr, device=device)
                    if t_wr.ndim == 3:
                        t_wr = t_wr.permute(2, 0, 1) if t_wr.shape[-1] == 3 else t_wr
                        t_wr = t_wr.unsqueeze(0)
                    obs_formatted["observation.images.wrist"] = t_wr.float()
                if ex is not None:
                    t_ex = torch.as_tensor(ex, device=device)
                    if t_ex.ndim == 3:
                        t_ex = t_ex.permute(2, 0, 1) if t_ex.shape[-1] == 3 else t_ex
                        t_ex = t_ex.unsqueeze(0)
                    obs_formatted["observation.images.extrinsic"] = t_ex.float()

                with torch.no_grad():
                    policy.reset()
                    obs_proc = preprocessor(obs_formatted)
                    act_pred = policy.select_action(obs_proc)
                    if postprocessor is not None:
                        act_pred = postprocessor(act_pred)
                    pred_action = act_pred.squeeze(0).cpu().numpy()

                mse = float(np.mean((pred_action - gt_action) ** 2))
                mses.append(mse)

            mse_score = float(np.mean(mses))
            print(f"  • Mean Action Prediction MSE : \033[1;36m{mse_score:.5f}\033[0m")
            if mse_score < 0.01:
                print("  • Status : \033[1;32m✓ PASS (High Accuracy Alignment)\033[0m")
            elif mse_score < 0.05:
                print("  • Status : \033[1;33m⚠ WARNING (Partial Convergence - keep training)\033[0m")
            else:
                print("  • Status : \033[1;31m❌ FAIL (Model Action Predictions Misaligned)\033[0m")
        except Exception as e:
            print(f"  • Could not run offline MSE check: {e}")
    else:
        print("  • Dataset root not found. Skipping Test #1.")

    # =========================================================================
    # TEST #2: FULL PICK-AND-PLACE STAGE VERIFICATION IN MUJOCO (300 STEPS)
    # =========================================================================
    print(f"\n\033[1;33m[Test #2] Running Full Pick-and-Place Task Verification ({trace_steps} steps)...\033[0m")
    env = PiperGymEnv(render_mode="rgb_array", wrist_cam_source="front", extrinsic_cam_source="scene")
    env.reset()

    # Match dataset starting pose
    start_qpos = np.array([0.0, -3.10, -0.25, 0.0, 0.0, 0.0, 0.04])  # HOME_QPOS default
    if dataset_root and "redcube_picknplace" in str(dataset_root).lower() and "100ep" not in str(dataset_root).lower() and "manual" not in str(dataset_root).lower():
        start_qpos = np.array([0.20, -2.00, -0.60, 0.00, 1.00, 0.00, 0.04])

    env.env.data.qpos[:7] = start_qpos
    env.env.data.ctrl[:7] = start_qpos
    mujoco.mj_forward(env.env.model, env.env.data)

    obs = env._get_obs(wrist_source="front")
    policy.reset()

    cube_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")
    bin_id = mujoco.mj_name2id(env.env.model, mujoco.mjtObj.mjOBJ_BODY, "bin_blue")

    ee_init = env.env.get_ee_pos()
    cube_init_pos = env.env.data.xpos[cube_id].copy()
    init_dist = float(np.linalg.norm(ee_init - cube_init_pos))

    print(f"  • Red Cube Spawn Position : [{cube_init_pos[0]:.3f}, {cube_init_pos[1]:.3f}, {cube_init_pos[2]:.3f}]")
    print(f"  • Initial EE Distance     : {init_dist * 100:.1f} cm")

    # Track 4 Stage Flags
    min_dist_to_cube = init_dist
    stage1_reached = False
    stage2_grasped = False
    stage3_lifted = False
    stage4_placed = False

    max_cube_height = cube_init_pos[2]
    final_cube_pos = cube_init_pos.copy()

    for step in range(trace_steps):
        st = obs["observation.state"]
        wr = obs["observation.images.wrist"]
        ex = obs["observation.images.extrinsic"]

        t_st = torch.as_tensor(st, dtype=torch.float32, device=device).unsqueeze(0)
        t_wr = torch.as_tensor(wr, device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        t_ex = torch.as_tensor(ex, device=device).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        obs_formatted = {
            "observation.state": t_st,
            "observation.images.wrist": t_wr,
            "observation.images.extrinsic": t_ex,
            "observation.images.camera1": t_wr,
            "observation.images.camera2": t_ex,
            "task": "pick up the red cube and place it into the blue bin",
        }

        with torch.no_grad():
            obs_proc = preprocessor(obs_formatted)
            act_tensor = policy.select_action(obs_proc)
            if postprocessor is not None:
                act_tensor = postprocessor(act_tensor)
            action = act_tensor.squeeze(0).cpu().numpy()

        res = env.step(action)
        obs = res[0]

        ee_pos = env.env.get_ee_pos()
        curr_cube_pos = env.env.data.xpos[cube_id].copy()
        dist = float(np.linalg.norm(ee_pos - curr_cube_pos))

        if dist < min_dist_to_cube:
            min_dist_to_cube = dist

        # Check Stage 1: Gripper within 3.5 cm of block
        if dist <= 0.035:
            stage1_reached = True

        # Check Stage 2: Gripper closes (ctrl <= 0.015) while near block
        if stage1_reached and action[6] <= 0.015:
            stage2_grasped = True

        # Check Stage 3: Red Cube lifted above initial table spawn height (+3 cm)
        if curr_cube_pos[2] > max_cube_height:
            max_cube_height = curr_cube_pos[2]
        if stage2_grasped and curr_cube_pos[2] >= cube_init_pos[2] + 0.03:
            stage3_lifted = True

        # Check Stage 4: Red Cube placed inside Blue Bin ([0.25 <= X <= 0.45], [0.25 <= Y <= 0.45])
        if stage3_lifted and (0.25 <= curr_cube_pos[0] <= 0.45) and (0.25 <= curr_cube_pos[1] <= 0.45) and curr_cube_pos[2] <= 0.18:
            stage4_placed = True

        final_cube_pos = curr_cube_pos

    print("\n  --- Full Task Stage Verification Summary ---")
    print(f"  • Min Distance to Cube : {min_dist_to_cube * 100:.1f} cm")
    print(f"  • Max Cube Height     : {max_cube_height * 100:.1f} cm (Table level: 2.0 cm)")

    s1_str = "\033[1;32m✓ PASS\033[0m" if stage1_reached else "\033[1;31m❌ FAIL\033[0m"
    s2_str = "\033[1;32m✓ PASS\033[0m" if stage2_grasped else "\033[1;31m❌ FAIL\033[0m"
    s3_str = "\033[1;32m✓ PASS\033[0m" if stage3_lifted else "\033[1;31m❌ FAIL\033[0m"
    s4_str = "\033[1;32m✓ PASS\033[0m" if stage4_placed else "\033[1;31m❌ FAIL\033[0m"

    print(f"  • Stage 1 (Reaching Block)     : {s1_str}")
    print(f"  • Stage 2 (Grasping & Finger)  : {s2_str}")
    print(f"  • Stage 3 (Lifting Off Table)  : {s3_str}")
    print(f"  • Stage 4 (Bin Placement)      : {s4_str}")

    if stage4_placed:
        print("  • OVERALL TASK STATUS          : \033[1;32m✓ FULL PICK-AND-PLACE SUCCESS!\033[0m")
    elif stage3_lifted:
        print("  • OVERALL TASK STATUS          : \033[1;33m⚠ PARTIAL SUCCESS (Lifted Cube, missed bin drop)\033[0m")
    elif stage1_reached:
        print("  • OVERALL TASK STATUS          : \033[1;33m⚠ PARTIAL SUCCESS (Reached Block, missed grasp/lift)\033[0m")
    else:
        print("  • OVERALL TASK STATUS          : \033[1;31m❌ FAILED TASK (Did not reach block)\033[0m")

    print("\033[1;34m========================================================================\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description="Agilex Piper Rapid Policy Verification Tool")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory or pretrained_model")
    parser.add_argument("--dataset-root", type=str, default=None, help="Optional path to dataset root for Test #1 MSE")
    parser.add_argument("--num-samples", type=int, default=5, help="Number of dataset frames to check for MSE")
    parser.add_argument("--steps", type=int, default=300, help="Number of simulation steps for Test #2 full rollout")
    args = parser.parse_args()

    verify_checkpoint(
        checkpoint_path=args.checkpoint,
        dataset_root=args.dataset_root,
        num_samples=args.num_samples,
        trace_steps=args.steps,
    )


if __name__ == "__main__":
    main()

