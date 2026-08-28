"""
Snapshot script — renders the scene from three cameras after settling physics.

  1. scene_cam   : overview of table, arm, both bins
  2. topdown_cam : straight down above source bin (clean object view)
  3. wrist_rgb   : wrist camera from arm hover pose (side-angle view)

Usage:
  python snapshot.py
  python snapshot.py --out /tmp/snaps
"""

import argparse
from pathlib import Path

import numpy as np
import mujoco
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.env import PiperEnv
from src.camera import WristCamera

# ---------------------------------------------------------------------------
# Arm hover pose: j2=2.87, j3=-1.5 keeps all links above the table.
# j5=1.2 tilts the wrist camera toward the workspace at ~45° angle.
# ---------------------------------------------------------------------------
HOVER_QPOS = {
    0: 0.94,   # joint1 — bearing toward source bin
    1: 2.87,   # joint2 — arm up and over
    2: -1.50,  # joint3 — elbow
    3: 0.00,   # joint4 — forearm roll
    4: 1.20,   # joint5 — wrist pitch (tilts camera toward workspace)
    5: 0.00,   # joint6
    6: 0.025,  # gripper
}


def render_camera(model, data, cam_name: str, height=600, width=800) -> np.ndarray:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer = mujoco.Renderer(model, height, width)
    renderer.update_scene(data, camera=cam_id)
    rgb = renderer.render().copy()
    renderer.close()
    return rgb


def render_depth(model, data, cam_name: str, height=480, width=640) -> np.ndarray:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer = mujoco.Renderer(model, height, width)
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=cam_id)
    depth = renderer.render().copy()
    renderer.close()
    return depth


def set_arm_pose(data, qpos_dict: dict) -> None:
    """Set arm joint qpos directly (bypasses physics — for rendering only)."""
    for idx, val in qpos_dict.items():
        data.qpos[idx] = val
    data.qvel[:] = 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="snapshots")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading environment…")
    env = PiperEnv(target_pos=np.array([0.42, 0.22, 0.31]))
    print("Settling scene (500 steps)…")
    env.data.ctrl[1] = -3.14
    env.data.ctrl[2] = 0.0
    for _ in range(500):
        mujoco.mj_step(env.model, env.data)

    # ── Shot 1: scene overview (arm at settled pose) ──────────────────────
    print("Rendering top-down view of source bin…")
    rgb_topdown = render_camera(env.model, env.data, "topdown_cam", height=540, width=540)
    dep_topdown = render_depth(env.model, env.data, "topdown_cam", height=540, width=540)

    # ── Shot 3: wrist camera from hover pose ──────────────────────────────
    print("Positioning arm for wrist camera shot…")
    set_arm_pose(env.data, HOVER_QPOS)
    mujoco.mj_fwdPosition(env.model, env.data)   # update kinematics only

    ee = env.get_ee_pos()
    print(f"  TCP at: {ee.round(3)}")

    cam = WristCamera(env.model, cam_name="wrist_rgb", height=480, width=640)
    rgb_wrist, dep_wrist = cam.get_rgb_and_depth(env.data)
    cam.close()

    # ── Compose figure ────────────────────────────────────────────────────
    fig = plt.figure(figsize=(22, 9), facecolor="#111827")

    layout = [
        # (subplot position, image, title, cmap, vmin, vmax)
        (231, rgb_scene,                  "Scene overview",        None,      None, None),
        (232, rgb_topdown,                "Top-down — source bin (RGB)", None, None, None),
        (233, np.clip(dep_topdown,0,1.0), "Top-down — depth (m)",  "plasma",  0,    1.0),
        (234, rgb_wrist,                  "Wrist camera — RGB",    None,      None, None),
        (235, np.clip(dep_wrist,0,0.8),   "Wrist camera — depth (m)", "plasma", 0, 0.8),
    ]

    for pos, img, title, cmap, vmin, vmax in layout:
        ax = fig.add_subplot(pos)
        ax.set_title(title, color="white", fontsize=10, pad=5)
        ax.axis("off")
        kwargs = dict(cmap=cmap)
        if vmin is not None:
            kwargs.update(vmin=vmin, vmax=vmax)
        im = ax.imshow(img, **kwargs)
        if cmap:
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.ax.tick_params(colors="white")
            plt.setp(cb.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle("Piper Arm — pick and place scene snapshot",
                 color="white", fontsize=14, y=1.01)
    fig.tight_layout()

    out_path = out_dir / "snapshot.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"\nSaved → {out_path}")
    env.close()


if __name__ == "__main__":
    main()
