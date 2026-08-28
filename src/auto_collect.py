"""
Automated Multi-Modal LeRobot Dataset Collector for Piper Arm Pick-and-Place.
-----------------------------------------------------------------------------
Autonomous closed-loop trajectory generation for picking the red cube and placing
it into the target blue bin with randomized tabletop jitter.

Features:
  - 100% automated demonstration collection (0 human effort required)
  - 4-camera multi-modal recording (Wrist RGB, Turbo Depth, Scene RGB, Topdown RGB)
  - Real-time physics stepping & 30 FPS LeRobot dataset capture
  - Parquet trajectories + MP4 videos + stats.json output
  - Automatic success verification & episode saving

Usage:
  python -m src.auto_collect --num-episodes 10 --data-dir data/red_block_dataset
  ./docker_run.sh --auto-collect --num-episodes 20 --data-dir data/red_block_dataset
"""

import sys
import time
import argparse
from pathlib import Path
import numpy as np
import mujoco
try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception:
    VIEWER_AVAILABLE = False

from src.environment.env import PiperEnv, HOME_QPOS, N_ARM_JOINTS, N_GRIPPER
from src.camera import WristCamera
from src.lerobot_dataset import LeRobotDatasetRecorder

# Constants
TARGET_BIN_POS = np.array([0.35, 0.32, 0.15])  # Target blue/teal bin center
TRANSIT_HEIGHT = 0.32
HOVER_HEIGHT = 0.28
GRASP_HEIGHT = 0.165
PLACE_HEIGHT = 0.22
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.00


class AutoPickAndPlaceAgent:
    """Autonomous trajectory controller for Piper red cube pick-and-place."""

    def __init__(self, env: PiperEnv):
        self.env = env
        self.model = env.model
        self.data = env.data

        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'ee')
        self.cube_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'cube_red')
        self.bin_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'target_bin')
        self.pad1_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'pad1')
        self.pad2_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'pad2')

    def get_pad_midpoint(self) -> np.ndarray:
        """Return the 3D center between the two gripper finger contact pads."""
        p1 = self.data.geom_xpos[self.pad1_id]
        p2 = self.data.geom_xpos[self.pad2_id]
        return (p1 + p2) / 2.0

    def get_cube_pos(self) -> np.ndarray:
        """Return current 3D position of the red cube."""
        return self.data.xpos[self.cube_id].copy()

    def step_ik(
        self,
        target_pad_pos: np.ndarray,
        target_dir: np.ndarray = np.array([0.0, 0.0, -1.0]),
        gripper_ctrl: float = GRIPPER_OPEN,
        step_scale: float = 0.08,
    ) -> np.ndarray:
        """Single Jacobian DLS IK step aligning pad midpoint and downward orientation."""
        curr_mid = self.get_pad_midpoint()
        curr_mat = self.data.site_xmat[self.site_id].reshape(3, 3)

        pos_err = target_pad_pos - curr_mid
        curr_z = curr_mat[:, 2]  # Gripper pointing axis (local Z)
        rot_err = np.cross(curr_z, target_dir)

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)

        J_pos = jacp[:, :6]
        J_rot = jacr[:, :6]

        err = np.concatenate([8.0 * pos_err, 2.0 * rot_err])
        J = np.vstack([J_pos, J_rot])

        damping = 1e-3
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)

        q_cmd = np.clip(
            self.data.qpos[:6] + dq * step_scale,
            self.model.jnt_range[:6, 0],
            self.model.jnt_range[:6, 1],
        )

        self.data.ctrl[:6] = q_cmd
        self.data.ctrl[6] = gripper_ctrl
        return q_cmd

    def is_cube_in_bin(self) -> bool:
        """Check if red cube is successfully inside the target blue bin."""
        cube = self.get_cube_pos()
        dx = abs(cube[0] - TARGET_BIN_POS[0])
        dy = abs(cube[1] - TARGET_BIN_POS[1])
        dz = abs(cube[2] - TARGET_BIN_POS[2])
        return dx < 0.10 and dy < 0.08 and dz < 0.12


def auto_collect_dataset(
    num_episodes: int = 10,
    data_dir: str = 'data/red_block_dataset',
    task_description: str = 'pick up the red cube and place it into the blue bin',
    fps: int = 30,
    headless: bool = False,
    viewer_sync: bool = True,
):
    print("\n" + "=" * 76)
    print("      Agilex Piper Automated LeRobot VLA Data Collection Pipeline")
    print("=" * 76)
    print(f"  Target Episodes : \033[1;32m{num_episodes}\033[0m")
    print(f"  Output Directory: \033[1;34m{data_dir}\033[0m")
    print(f"  Task Prompt     : \033[1;37m\"{task_description}\"\033[0m")
    print(f"  Capture Rate    : \033[1;36m{fps} FPS\033[0m")
    print(f"  Headless Mode   : \033[1;33m{headless}\033[0m")
    print("=" * 76 + "\n")

    env = PiperEnv()
    agent = AutoPickAndPlaceAgent(env)

    # Initialize 4 Cameras
    wrist_cam = WristCamera(env.model, "wrist_rgb", exposure=1.0)
    scene_cam = WristCamera(env.model, "scene_cam", exposure=1.0)
    topdown_cam = WristCamera(env.model, "topdown_cam", exposure=1.0)

    # Initialize LeRobot Dataset Recorder
    recorder = LeRobotDatasetRecorder(
        dataset_dir=data_dir,
        fps=fps,
        task_description=task_description,
    )

    viewer = None
    if not headless and VIEWER_AVAILABLE:
        try:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 1.15
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -22
        except Exception as e:
            print(f"\033[1;33mNote: Running headless (passive viewer disabled: {e})\033[0m")
            viewer = None

    substeps = 5  # 5 * 2ms = 10ms per control step

    successful_episodes = 0
    attempt = 0

    try:
        while successful_episodes < num_episodes:
            attempt += 1
            print(f"\n\033[1;34m========================================================================\033[0m")
            print(f"\033[1;34m[Episode {successful_episodes + 1}/{num_episodes}] (Attempt #{attempt})\033[0m")

            # 1. Reset Environment with Tabletop Cube Randomization
            env.reset(randomize_cubes=True)

            # Start in natural forward-reaching ready posture
            env.data.qpos[:6] = np.array([0.20, -2.00, -0.60, 0.00, 1.00, 0.00])
            env.data.ctrl[:6] = env.data.qpos[:6]
            env.data.ctrl[6] = GRIPPER_OPEN
            mujoco.mj_forward(env.model, env.data)

            if viewer is not None:
                viewer.sync()

            cube_init_pos = agent.get_cube_pos()
            print(f"  • Red Cube Spawn Position: [{cube_init_pos[0]:.4f}, {cube_init_pos[1]:.4f}, {cube_init_pos[2]:.4f}]")

            # Start LeRobot Recording
            recorder.start_recording(task_description=task_description)

            def record_and_step(target_pos: np.ndarray, gripper_val: float, steps: int):
                for _ in range(steps):
                    # Capture Multi-Modal Step
                    state = np.concatenate([env.data.qpos[:N_ARM_JOINTS], [env.data.qpos[N_ARM_JOINTS]]])
                    action = np.concatenate([env.data.ctrl[:N_ARM_JOINTS], [gripper_val]])

                    w_rgb, w_dep = wrist_cam.get_rgb_and_depth(env.data)
                    s_rgb = scene_cam.get_rgb(env.data)
                    t_rgb = topdown_cam.get_rgb(env.data)

                    recorder.record_step(state, action, w_rgb, w_dep, s_rgb, t_rgb)

                    # Compute IK and step physics
                    agent.step_ik(target_pos, gripper_ctrl=gripper_val)
                    for _ in range(substeps):
                        mujoco.mj_step(env.model, env.data)

                    if viewer is not None and viewer_sync:
                        viewer.sync()

            # -------------------------------------------------------------
            # Autonomous Trajectory Execution
            # -------------------------------------------------------------
            # Waypoint 1: Approach & Hover above Cube (open gripper)
            hover_pos = np.array([cube_init_pos[0], cube_init_pos[1], HOVER_HEIGHT])
            record_and_step(hover_pos, GRIPPER_OPEN, steps=35)

            # Waypoint 2: Descend to Grasp Level
            grasp_pos = np.array([cube_init_pos[0], cube_init_pos[1], GRASP_HEIGHT])
            record_and_step(grasp_pos, GRIPPER_OPEN, steps=30)

            # Waypoint 3: Clamp Gripper firmly on Cube
            record_and_step(grasp_pos, GRIPPER_CLOSED, steps=30)

            # Waypoint 4: Lift Cube smoothly
            lift_pos = np.array([cube_init_pos[0], cube_init_pos[1], TRANSIT_HEIGHT])
            record_and_step(lift_pos, GRIPPER_CLOSED, steps=30)

            # Waypoint 5: Transport over to Target Blue Bin
            bin_above_pos = np.array([TARGET_BIN_POS[0], TARGET_BIN_POS[1], TRANSIT_HEIGHT])
            record_and_step(bin_above_pos, GRIPPER_CLOSED, steps=45)

            # Waypoint 6: Lower into Blue Bin
            bin_place_pos = np.array([TARGET_BIN_POS[0], TARGET_BIN_POS[1], PLACE_HEIGHT])
            record_and_step(bin_place_pos, GRIPPER_CLOSED, steps=25)

            # Waypoint 7: Open Gripper to Release Cube
            record_and_step(bin_place_pos, GRIPPER_OPEN, steps=25)

            # Waypoint 8: Ascend back to Transit Height
            record_and_step(bin_above_pos, GRIPPER_OPEN, steps=25)

            # Waypoint 9: Settle and Return towards Ready Posture
            ready_pos = np.array([0.28, 0.12, TRANSIT_HEIGHT])
            record_and_step(ready_pos, GRIPPER_OPEN, steps=20)

            # -------------------------------------------------------------
            # Success Verification & Commitment
            # -------------------------------------------------------------
            if agent.is_cube_in_bin():
                successful_episodes += 1
                final_cube = agent.get_cube_pos()
                print(f"  \033[1;32m✓ Pick & Place SUCCESS! (Cube in bin at [{final_cube[0]:.3f}, {final_cube[1]:.3f}, {final_cube[2]:.3f}])\033[0m")
                recorder.save_episode(success=True)
            else:
                print(f"  \033[1;31m✗ Attempt FAILED (Cube missed bin). Discarding episode.\033[0m")
                recorder.discard_current_episode()

    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass

    print(f"\n\033[1;32m========================================================================")
    print(f"  ✓ Autonomous Data Collection Complete!")
    print(f"  Recorded {successful_episodes} verified episodes into \"{data_dir}\"")
    print(f"========================================================================\033[0m\n")


def main():
    parser = argparse.ArgumentParser(description='Autonomous LeRobot VLA Pick-and-Place Data Collection')
    parser.add_argument('--num-episodes', type=int, default=10,
                        help='Number of successful episodes to record (default: 10)')
    parser.add_argument('--data-dir', type=str, default='data/red_block_dataset',
                        help='Target dataset directory (default: data/red_block_dataset)')
    parser.add_argument('--task', type=str, default='pick up the red cube and place it into the blue bin',
                        help='Language instruction prompt for VLA training')
    parser.add_argument('--fps', type=int, default=30,
                        help='Recording FPS (default: 30)')
    parser.add_argument('--headless', action='store_true', default=False,
                        help='Run in headless mode without opening GUI window')
    args = parser.parse_args()

    auto_collect_dataset(
        num_episodes=args.num_episodes,
        data_dir=args.data_dir,
        task_description=args.task,
        fps=args.fps,
        headless=args.headless,
    )


if __name__ == '__main__':
    main()
