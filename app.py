"""
Piper 6-DOF Pure TCP Cartesian Teleoperation & LeRobot VLA Data Collection Pipeline.
-------------------------------------------------------------------------------------
Control Scheme: Decoupled TCP Cartesian Teleoperation
  - Workspace Translation : Forward/Back (X), Left/Right (Y), Elevation (Z)
  - Tool-Frame Rotation   : Roll (Wrist twist), Pitch (Wrist nod), Yaw (Wrist pan)
  - Analog Gripper        : Smooth continuous rate positional control (0..40 mm)
  - Multi-Camera & Graph  : Real-time 640x720 4-camera visualizer + live telemetry graph
  - LeRobot VLA Recorder  : Lossless multi-modal data collection with tqdm progress
"""

import sys
import time
import argparse
import select
import termios
import tty
from typing import Optional
import numpy as np
import mujoco
import mujoco.viewer

try:
    import av
    av.logging.set_level(av.logging.ERROR)
except Exception:
    pass

from src.environment.env import PiperEnv
from src.controllers.ik_controller import DifferentialIKController, mat2euler
from src.spacemouse import SpaceMouse
from src.camera import WristCamera
from src.camera_visualizer import MultiCameraVisualizer
from src.telemetry_plotter import TelemetryGraphPlotter
from src.lerobot_dataset import LeRobotDatasetRecorder

# Constants & Limits
N_ARM_JOINTS = 6
HOME_QPOS = np.array([0.0, -3.14, -0.22, 0.0, 0.0, 0.0, 0.0, 0.0])
GRIPPER_MIN = 0.00
GRIPPER_MAX = 0.04
CAMERA_EVERY = 5


class RawKeyboard:
    """Non-blocking single character terminal keyboard reader."""

    def __enter__(self):
        self.old_settings = None
        if sys.stdin.isatty():
            try:
                self.old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except Exception:
                pass
        return self

    def __exit__(self, *args):
        if self.old_settings is not None:
            try:
                termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
            except Exception:
                pass

    def get_char(self) -> Optional[str]:
        if not sys.stdin.isatty():
            return None
        try:
            r, _, _ = select.select([sys.stdin], [], [], 0.0)
            if r:
                return sys.stdin.read(1)
        except Exception:
            pass
        return None


def print_banner(sm: SpaceMouse, task_description: str, data_dir: str):
    """Print the complete interactive control menu banner at startup."""
    print("\n" + "=" * 76)
    print("       Agilex Piper 6-DOF Pure TCP Teleoperation & LeRobot VLA Pipeline")
    print("=" * 76)
    backend_name = "MuJoCo Built-in Analytical Kinematics"
    print(f"  Backend Engine : \033[1;32m{backend_name}\033[0m")
    if sm.is_connected:
        print(f"  SpaceMouse     : \033[1;32mCONNECTED ({sm.device_name})\033[0m")
        print("                   \033[1;36mLeft Button: Close Gripper | Right Button: Open Gripper\033[0m")
        print("                   \033[1;33mBoth Buttons Held: Reset Home Pose & Table Objects\033[0m")
    else:
        print("  SpaceMouse     : \033[1;33mNot Detected (Keyboard Active)\033[0m")
    print("  Control Scheme : \033[1;36mDecoupled (Workspace Translation + Tool Rotation)\033[0m")
    print(f"  Dataset Target : \033[1;34m{data_dir}\033[0m")
    print(f"  Task Prompt    : \033[1;37m\"{task_description}\"\033[0m")
    print("-" * 76)
    print("  [W / S]       — Forward / Backward  (along table workspace +X / -X)")
    print("  [A / D]       — Left / Right        (across table workspace +Y / -Y)")
    print("  [R / F]       — Elevation Height    (Up / Down in room Z axis +Z / -Z)")
    print("  [U / O]       — Roll  ±             (wrist twist around pointing axis)")
    print("  [I / K]       — Pitch ±             (wrist nod up / down)")
    print("  [J / L]       — Yaw   ±             (wrist pan left / right)")
    print("  [P]           — Toggle SpaceMouse   [ON / PAUSED]")
    print("  [ [ / ] ]     — Open / Close Gripper Position")
    print("  [C / Space]   — Start / Stop & Save LeRobot Episode Recording")
    print("  [N]           — Discard Current Recording Buffer")
    print("  [1 / 2 / 3]   — Speed Mode (Fine / Normal / Fast)")
    print("  [H]           — Reset to Home Pose & Re-randomize Cubes")
    print("  [Q / ESC]     — Quit teleoperation")
    print("=" * 76 + "\n")


def print_status(
    env: PiperEnv,
    gripper_ctrl: float,
    speed_mode: str,
    sm_enabled: bool,
    sm_connected: bool,
    recorder: LeRobotDatasetRecorder,
):
    """Print continuous real-time single-line status."""
    ee_pos = env.get_ee_pos()
    qpos = env.data.qpos[:N_ARM_JOINTS]
    sm_status = "ON" if (sm_enabled and sm_connected) else ("PAUSED" if sm_connected else "N/A")
    rec_status = (
        f"\033[1;31mREC Ep {recorder.num_episodes:03d}\033[0m"
        if recorder.is_recording
        else "\033[1;30mIDLE\033[0m"
    )

    status_str = (
        f"\r\033[K[Piper TCP] "
        f"Pos=[{ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, {ee_pos[2]:+.3f}]m | "
        f"Grip: {gripper_ctrl*1000.0:4.1f}mm | "
        f"SM: {sm_status:6s} | "
        f"Spd: {speed_mode:6s} | "
        f"Status: {rec_status}"
    )
    sys.stdout.write(status_str)
    sys.stdout.flush()


def run(
    env: PiperEnv,
    show_camera: bool = True,
    data_dir: str = "data/lerobot_dataset",
    task_description: str = "pick up the red cube and place it into the bin",
    exposure: float = 1.0,
) -> None:
    sm = SpaceMouse()
    sm.start()
    sm_enabled = sm.is_connected

    print_banner(sm, task_description, data_dir)

    gripper_ctrl = GRIPPER_MIN
    ik = DifferentialIKController(env.model, site_name="ee", home_qpos=HOME_QPOS[:6])

    recorder = LeRobotDatasetRecorder(
        dataset_dir=data_dir,
        fps=30,
        task_description=task_description,
        image_height=480,
        image_width=640,
    )

    wrist_cam = WristCamera(env.model, cam_name="wrist_rgb", height=480, width=640, exposure=exposure)
    scene_cam = WristCamera(env.model, cam_name="scene_cam", height=480, width=640, exposure=exposure)
    front_cam = WristCamera(env.model, cam_name="front_cam", height=480, width=640, exposure=exposure)

    telemetry_plotter = TelemetryGraphPlotter(width=640, height=240, max_history=150)
    cam_viz = MultiCameraVisualizer(include_graph=True) if show_camera else None

    speed_modes = {
        "1": ("Fine", 0.003, 0.03),
        "2": ("Normal", 0.008, 0.06),
        "3": ("Fast", 0.020, 0.12),
    }
    cur_speed_key = "2"
    pos_step, rot_step = speed_modes[cur_speed_key][1:]

    step = 0
    kb = RawKeyboard()
    viewer = None
    last_record_time = 0.0
    dt = env.model.opt.timestep
    gripper_speed = 0.025  # 25 mm/s smooth rate control

    try:
        with kb:
            viewer = mujoco.viewer.launch_passive(env.model, env.data)
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 140
            viewer.cam.elevation = -22

            def do_reset():
                nonlocal gripper_ctrl
                env.reset()
                ik.reset(env.data.qpos[:N_ARM_JOINTS])
                telemetry_plotter.reset()
                init_state = np.concatenate([env.data.qpos[:N_ARM_JOINTS], [env.data.qpos[N_ARM_JOINTS]]])
                telemetry_plotter.add_sample(init_state)
                gripper_ctrl = GRIPPER_MIN
                env.data.ctrl[N_ARM_JOINTS] = gripper_ctrl
                viewer.sync()

            do_reset()

            while viewer.is_running():
                # -------------------------------------------------------------
                # 1. SpaceMouse Continuous 6-DOF & Positional Gripper Control
                # -------------------------------------------------------------
                if sm_enabled and sm.is_connected:
                    sx, sy, sz, spitch, syaw, sroll = sm.get_axes()
                    b_left, b_right = sm.get_buttons()

                    # Continuous Position Control: hold button to glide to exact position
                    if b_left == 1 and b_right == 1:
                        do_reset()
                    elif b_left == 1:
                        gripper_ctrl = max(gripper_ctrl - gripper_speed * dt, GRIPPER_MIN)
                    elif b_right == 1:
                        gripper_ctrl = min(gripper_ctrl + gripper_speed * dt, GRIPPER_MAX)

                    # Simultaneous 6-DOF teleoperation
                    if any(abs(v) > 0.01 for v in (sx, sy, sz, spitch, syaw, sroll)):
                        scale_p = pos_step * 0.20
                        scale_r = rot_step * 0.35

                        # Forward (+X, flipped front/back), Lateral (+Y), Elevation (+Z)
                        d_pos = np.array([-sy * scale_p, sx * scale_p, sz * scale_p])
                        # Tool Rotation: Swap Yaw and Roll, with flipped Yaw direction
                        d_rpy = np.array([syaw * scale_r, spitch * scale_r, -sroll * scale_r])

                        q_target = ik.solve_decoupled(
                            env.data.qpos[:N_ARM_JOINTS],
                            env.get_ee_pos(),
                            env.get_ee_mat(),
                            d_pos,
                            d_rpy,
                        )
                        env.data.ctrl[:N_ARM_JOINTS] = q_target

                # -------------------------------------------------------------
                # 2. Keyboard Control (Fallback / Fine Adjustment)
                # -------------------------------------------------------------
                key = kb.get_char()
                if key is not None:
                    key = key.lower()
                    d_pos = np.zeros(3)
                    d_rpy = np.zeros(3)

                    if key in ("\x1b", "q"):
                        break
                    elif key == "w":
                        d_pos[0] += pos_step
                    elif key == "s":
                        d_pos[0] -= pos_step
                    elif key == "a":
                        d_pos[1] += pos_step
                    elif key == "d":
                        d_pos[1] -= pos_step
                    elif key == "r":
                        d_pos[2] += pos_step
                    elif key == "f":
                        d_pos[2] -= pos_step
                    elif key == "u":
                        d_rpy[0] += rot_step
                    elif key == "o":
                        d_rpy[0] -= rot_step
                    elif key == "i":
                        d_rpy[1] += rot_step
                    elif key == "k":
                        d_rpy[1] -= rot_step
                    elif key == "j":
                        d_rpy[2] += rot_step
                    elif key == "l":
                        d_rpy[2] -= rot_step
                    elif key == "[":
                        gripper_ctrl = min(gripper_ctrl + 0.005, GRIPPER_MAX)
                    elif key == "]":
                        gripper_ctrl = max(gripper_ctrl - 0.005, GRIPPER_MIN)
                    elif key in ("c", " "):
                        if not recorder.is_recording:
                            recorder.start_recording(task_description=task_description)
                        else:
                            recorder.save_episode(success=True)
                            do_reset()
                    elif key == "n":
                        recorder.discard_episode()
                        do_reset()
                    elif key == "p":
                        sm_enabled = not sm_enabled
                    elif key == "h":
                        do_reset()
                    elif key in ("1", "2", "3"):
                        cur_speed_key = key
                        pos_step, rot_step = speed_modes[key][1:]

                    if np.any(d_pos != 0) or np.any(d_rpy != 0):
                        q_target = ik.solve_decoupled(
                            env.data.qpos[:N_ARM_JOINTS],
                            env.get_ee_pos(),
                            env.get_ee_mat(),
                            d_pos,
                            d_rpy,
                        )
                        env.data.ctrl[:N_ARM_JOINTS] = q_target

                # Step physics
                env.data.ctrl[N_ARM_JOINTS] = gripper_ctrl
                mujoco.mj_step(env.model, env.data)

                # -------------------------------------------------------------
                # 3. LeRobot Step Capture (30 Hz)
                # -------------------------------------------------------------
                now = time.time()
                state = np.concatenate([env.data.qpos[:N_ARM_JOINTS], [env.data.qpos[N_ARM_JOINTS]]])
                action = np.concatenate([env.data.ctrl[:N_ARM_JOINTS], [gripper_ctrl]])

                if recorder.is_recording and (now - last_record_time >= (1.0 / recorder.fps)):
                    last_record_time = now
                    w_rgb = wrist_cam.get_rgb(env.data)
                    s_rgb = scene_cam.get_rgb(env.data)
                    f_rgb = front_cam.get_rgb(env.data)
                    recorder.record_step(
                        state=state,
                        action=action,
                        wrist_rgb=w_rgb,
                        extrinsic_rgb=s_rgb,
                        topdown_rgb=f_rgb,
                    )

                # -------------------------------------------------------------
                # 4. Multi-Camera & Telemetry Graph Visualizer Window
                # -------------------------------------------------------------
                if step % CAMERA_EVERY == 0:
                    telemetry_plotter.add_sample(state)
                    if cam_viz is not None and cam_viz.is_open:
                        w_rgb = wrist_cam.get_rgb(env.data)
                        s_rgb = scene_cam.get_rgb(env.data)
                        f_rgb = front_cam.get_rgb(env.data)
                        graph_img = telemetry_plotter.render_live_graph()
                        cam_viz.update(w_rgb, None, s_rgb, f_rgb, graph_rgb=graph_img)

                    print_status(
                        env,
                        gripper_ctrl,
                        speed_modes[cur_speed_key][0],
                        sm_enabled=sm_enabled,
                        sm_connected=sm.is_connected,
                        recorder=recorder,
                    )

                viewer.sync()
                step += 1
                time.sleep(dt)

    finally:
        sm.stop()
        if recorder.is_recording:
            recorder.save_episode(success=True)
        recorder.finalize()
        if viewer is not None:
            try:
                viewer.close()
            except Exception:
                pass
        if cam_viz is not None:
            cam_viz.close()
        sys.stdout.write("\r\033[K\nTeleoperation stopped. Goodbye!\n")
        sys.stdout.flush()


def main() -> None:
    parser = argparse.ArgumentParser(description="Piper Arm MuJoCo Teleoperation & LeRobot VLA Data Collection")
    parser.add_argument("--target", nargs=3, type=float, metavar=("X", "Y", "Z"),
                        default=[0.42, 0.22, 0.31],
                        help="Goal marker position (default: 0.42 0.22 0.31)")
    parser.add_argument("--camera", action="store_true", default=True,
                        help="Show live multi-camera feedback visualizer window (default: True)")
    parser.add_argument("--no-camera", action="store_false", dest="camera",
                        help="Disable multi-camera visualizer window")
    parser.add_argument("--data-dir", type=str, default="data/lerobot_dataset",
                        help="Directory to store recorded LeRobot dataset (default: data/lerobot_dataset)")
    parser.add_argument("--task", type=str, default="pick up the red cube and place it into the bin",
                        help="Language task instruction description for VLA training")
    parser.add_argument("--exposure", type=float, default=1.0,
                        help="Camera brightness/exposure gain multiplier (default: 1.0, e.g. 1.3 to brighten)")
    args = parser.parse_args()

    target = np.array(args.target)
    env = PiperEnv(target_pos=target)

    print(f"Piper arm loaded  |  target={target}  |  ee_start={env.get_ee_pos()}")

    try:
        run(
            env,
            show_camera=args.camera,
            data_dir=args.data_dir,
            task_description=args.task,
            exposure=args.exposure,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        env.close()


if __name__ == "__main__":
    main()
