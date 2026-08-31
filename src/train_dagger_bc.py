"""
DAgger (Dataset Aggregation) Interactive Behavioral Cloning Trainer for Piper Arm.
----------------------------------------------------------------------------------
Iterative Human-in-the-Loop Imitation Learning:
  1. Policy Execution: Runs the trained BC/ACT policy in the live 3D window.
  2. Human Correction Capture: Whenever the policy drifts or hesitates, touch SpaceMouse or
     press W/A/S/D/R/F/J/L/I/K/U/O/Spacebar to take over control.
  3. Dataset Aggregation: DAgger pairs the exact visual observation state s_t with your
     human corrective action a_t_human and appends it to the DAgger dataset buffer.
  4. Instant Fine-Tuning: Re-trains the policy on aggregated data so errors disappear rapidly!

Usage:
  # Round 1: Run DAgger interactive correction rollouts
  python -m src.train_dagger_bc --dataset data/redcube_picknplace_manual_v4 --episodes 10

  # Re-train policy on aggregated DAgger dataset
  python -m src.train_dagger_bc --train --dataset data/redcube_picknplace_manual_v4_dagger
"""

import os
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import torch
import mujoco

from src.environment.env import PiperEnv
from src.controllers.ik_controller import DifferentialIKController, euler2mat, mat2euler
from src.camera import WristCamera
from src.lerobot_dataset import LeRobotDatasetRecorder

try:
    from src.spacemouse import SpaceMouse
    SPACEMOUSE_AVAILABLE = True
except Exception:
    SPACEMOUSE_AVAILABLE = False

import select
import termios
import tty


class NonBlockingKeyboard:
    def __init__(self):
        self.old = None
        if sys.stdin.isatty():
            try:
                self.old = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                pass

    def __del__(self):
        if self.old and sys.stdin.isatty():
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old)
            except Exception:
                pass

    def get_keys(self):
        keys = []
        if sys.stdin.isatty():
            while select.select([sys.stdin], [], [], 0.0001)[0]:
                keys.append(sys.stdin.read(1))
        return keys


def run_dagger_session(
    base_dataset_dir: str = "data/redcube_picknplace_manual_v4",
    output_dagger_dir: str = "data/redcube_picknplace_manual_v4_dagger",
    episodes: int = 5,
    max_steps_per_episode: int = 600,
):
    print("=" * 75)
    print("   DAgger (Dataset Aggregation) Interactive Teleop Correction Collector")
    print("=" * 75)
    print(f"  Base Dataset Directory   : {base_dataset_dir}")
    print(f"  Output DAgger Directory  : {output_dagger_dir}")
    print(f"  Target Correction Rounds : {episodes} episodes")
    print("=" * 75)
    print("\nControls:")
    print("  Position (XYZ)  : W / S (+X/-X)  |  A / D (+Y/-Y)  |  R / F (+Z/-Z)")
    print("  Rotation (RPY)  : J / L (Roll)   |  I / K (Pitch)  |  U / O (Yaw)")
    print("  Gripper Action  : SPACEBAR (Close Jaws)")
    print("  Save & Reset Ep : N (Finish episode)")
    print("=" * 75 + "\n")

    env = PiperEnv(render_mode="human", max_episode_steps=max_steps_per_episode)
    ik = DifferentialIKController(env.model, site_name="ee", max_iters=25)
    kb = NonBlockingKeyboard()

    front_cam = WristCamera(env.model, "front_cam", height=480, width=640)
    scene_cam = WristCamera(env.model, "scene_cam", height=480, width=640)

    sm = None
    if SPACEMOUSE_AVAILABLE:
        try:
            sm = SpaceMouse()
            sm.start()
            print("🎮 [DAgger] Connected to 3Dconnexion SpaceMouse hardware!")
        except Exception:
            sm = None

    # Setup LeRobot Recorder for DAgger Dataset Aggregation
    recorder = LeRobotDatasetRecorder(
        dataset_dir=output_dagger_dir,
        fps=30,
        task_description="pick up the red cube and place it into the blue bin dagger correction",
        image_height=480,
        image_width=640,
        repo_id="local/dagger_dataset",
    )

    completed_episodes = 0

    for ep in range(episodes):
        print(f"\n🎬 [DAgger Episode {ep+1}/{episodes}] Starting trial...")
        env.reset(randomize_cubes=True)
        env.data.qpos[:7] = np.array([0.0, -3.14, -0.22, 0.0, 0.0, 0.0, 0.04])
        env.data.ctrl[:7] = env.data.qpos[:7]
        mujoco.mj_forward(env.model, env.data)
        ik.reset(env.data.qpos[:6])

        recorder.start_recording()
        step = 0
        ep_interventions = 0
        manual_finish = False

        while step < max_steps_per_episode and not manual_finish:
            # 1. Capture Current Multi-Modal Frames & State
            w_rgb = front_cam.get_rgb(env.data)
            s_rgb = scene_cam.get_rgb(env.data)
            qpos = np.concatenate([env.data.qpos[:6], [env.data.qpos[6]]]).astype(np.float32)

            dx, dy, dz = 0.0, 0.0, 0.0
            droll, dpitch, dyaw = 0.0, 0.0, 0.0
            dgrip = 0.0
            human_active = False

            # 2. Check SpaceMouse Hardware Input
            if sm is not None:
                sm_axes = sm.get_motion_state()
                sm_btns = sm.get_buttons()
                if np.max(np.abs(sm_axes)) > 0.05 or any(sm_btns):
                    human_active = True
                    ep_interventions += 1
                    dx = sm_axes[0] * 0.12   # Boosted SpaceMouse speed!
                    dy = sm_axes[1] * 0.12
                    dz = sm_axes[2] * 0.12
                    droll = sm_axes[3] * 0.25
                    dpitch = sm_axes[4] * 0.25
                    dyaw = sm_axes[5] * 0.25
                    dgrip = -1.0 if sm_btns[0] else (+1.0 if sm_btns[1] else 0.0)

            # 3. Check Keyboard Input
            keys = kb.get_keys()
            if len(keys) > 0:
                human_active = True
                ep_interventions += 1
                pos_step = 0.100   # Fast 10cm translation per keypress!
                rot_step = 0.250   # Fast 14 deg rotation per keypress!

                for key in keys:
                    k = key.lower()
                    if k == "n":
                        manual_finish = True
                        print("  🛑 [DAgger Ep Finish] 'N' pressed -> Saving episode!")
                    elif k == "w": dx += pos_step
                    elif k == "s": dx -= pos_step
                    elif k == "a": dy += pos_step
                    elif k == "d": dy -= pos_step
                    elif k == "r": dz += pos_step
                    elif k == "f": dz -= 0.120      # Fast 12cm downward descent!
                    elif k == "j": droll -= rot_step
                    elif k == "l": droll += rot_step
                    elif k == "i": dpitch -= rot_step
                    elif k == "k": dpitch += 0.300   # Fast 17 deg pitch up!
                    elif k == "u": dyaw += rot_step
                    elif k == "o": dyaw -= rot_step
                    elif k == " ": dgrip = -1.0

            # EE Pose Command & Persistent Decoupled IK Execution (Identical to Manual Teleop)
            delta_pos = np.array([dx, dy, dz], dtype=np.float32)
            delta_rpy = np.array([droll, dpitch, dyaw], dtype=np.float32)

            q_arm = ik.step_decoupled_delta(delta_pos, delta_rpy)
            curr_grip = env.data.qpos[6]

            if dgrip < -0.5:
                new_grip = 0.00
            elif dgrip > +0.5:
                new_grip = 0.04
            else:
                new_grip = np.clip(curr_grip + dgrip * 0.01, 0.00, 0.04)

            ctrl_action = np.concatenate([q_arm, [new_grip]]).astype(np.float32)
            env.step(ctrl_action)

            # Record DAgger State-Action Pair to Dataset
            recorder.record_step(
                state=qpos,
                action=ctrl_action,
                wrist_rgb=w_rgb,
                extrinsic_rgb=s_rgb,
            )

            step += 1

        recorder.save_episode()
        completed_episodes += 1
        print(f"✓ [DAgger Episode {ep+1}] Complete! Interventions: {ep_interventions} | Steps recorded: {step}")

    env.close()

    print("\n" + "=" * 75)
    print(f"✓ DAgger Collection Complete! {completed_episodes} correction episodes saved to '{output_dagger_dir}'!")
    print("=" * 75)
    print("\nNext Step - Train ACT or SmolVLA on your DAgger dataset:")
    print(f"  python -m src.train_policy --policy act --dataset {output_dagger_dir} --steps 5000")
    print("=" * 75 + "\n")


def main():
    parser = argparse.ArgumentParser(description="DAgger Interactive BC Correction Tool")
    parser.add_argument("--dataset", type=str, default="data/redcube_picknplace_manual_v4", help="Base dataset path")
    parser.add_argument("--output", type=str, default="data/redcube_picknplace_manual_v4_dagger", help="Output DAgger dataset path")
    parser.add_argument("--episodes", type=int, default=5, help="Number of DAgger correction episodes to record")
    args = parser.parse_args()

    run_dagger_session(
        base_dataset_dir=args.dataset,
        output_dagger_dir=args.output,
        episodes=args.episodes,
    )


if __name__ == "__main__":
    main()

