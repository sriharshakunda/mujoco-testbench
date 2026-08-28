"""
Renders a video of the Piper arm:
  - scene_cam overview (arm + both bins)
  - Sequence: home → hover over bin → gripper open/close → return home
Saves an MP4 (or GIF fallback) to ./videos/piper_demo.mp4

Usage:
  python make_video.py
  python make_video.py --fps 30 --out videos/demo.mp4
"""

import argparse
from pathlib import Path

import numpy as np
import mujoco
import imageio

from src.environment.env import PiperEnv

# ---------------------------------------------------------------------------
# Camera helper
# ---------------------------------------------------------------------------

def render_frame(renderer, model, data, cam_name: str) -> np.ndarray:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    renderer.update_scene(data, camera=cam_id)
    return renderer.render().copy()


# ---------------------------------------------------------------------------
# Motion keyframes  (qpos[:7] = joint1-6 + gripper)
# ---------------------------------------------------------------------------

HOME    = np.array([0.00,  0.00,  0.00,  0.00,  0.00,  0.00,  0.00])
HOVER   = np.array([0.94,  2.87, -1.50,  0.00,  1.20,  0.00,  0.00])
OPEN    = np.array([0.94,  2.87, -1.50,  0.00,  1.20,  0.00,  0.045])
CLOSED  = np.array([0.94,  2.87, -1.50,  0.00,  1.20,  0.00,  0.00])


def lerp(a, b, t):
    return a + (b - a) * np.clip(t, 0, 1)


def make_sequence(fps: int):
    """Return list of (ctrl, n_physics_steps) for each keyframe segment."""
    step_dt = 0.002                    # model timestep
    def dur(seconds): return int(seconds / step_dt)

    # Build segments: (start_ctrl, end_ctrl, physics_steps, render_every)
    render_every = max(1, int((1 / fps) / step_dt))   # steps between frames

    segments = [
        (HOME,   HOME,   dur(1.5), render_every),   # pause at home
        (HOME,   HOVER,  dur(3.0), render_every),   # move to hover
        (HOVER,  HOVER,  dur(0.5), render_every),   # pause
        (HOVER,  OPEN,   dur(1.5), render_every),   # open gripper
        (OPEN,   OPEN,   dur(0.8), render_every),   # hold open
        (OPEN,   CLOSED, dur(1.5), render_every),   # close gripper
        (CLOSED, CLOSED, dur(0.5), render_every),   # pause
        (CLOSED, HOME,   dur(3.0), render_every),   # return home
        (HOME,   HOME,   dur(1.0), render_every),   # pause at home
    ]
    return segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps",  type=int,   default=30)
    parser.add_argument("--out",  default="videos/piper_demo.mp4")
    parser.add_argument("--cam",  default="scene_cam",
                        help="Camera name (scene_cam | topdown_cam | wrist_rgb)")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Loading environment…")
    env = PiperEnv()
    env.reset()

    print("Settling spheres (3000 steps)…")
    for _ in range(3000):
        mujoco.mj_step(env.model, env.data)

    renderer = mujoco.Renderer(env.model, height=540, width=800)

    segments   = make_sequence(args.fps)
    step_dt    = env.model.opt.timestep
    render_every = max(1, int((1 / args.fps) / step_dt))

    frames = []
    total_steps = sum(s[2] for s in segments)
    done = 0

    print(f"Rendering {total_steps} physics steps → ~{args.fps} fps video…")
    for start_ctrl, end_ctrl, n_steps, _ in segments:
        for i in range(n_steps):
            t = i / max(n_steps - 1, 1)
            env.data.ctrl[:7] = lerp(start_ctrl, end_ctrl, t)
            mujoco.mj_step(env.model, env.data)
            done += 1

            if i % render_every == 0:
                frames.append(render_frame(renderer, env.model, env.data, args.cam))

        if done % 500 == 0 or done == total_steps:
            print(f"  {done}/{total_steps} steps  ({len(frames)} frames)")

    renderer.close()
    env.close()

    print(f"Writing {len(frames)} frames → {out_path} …")
    ext = out_path.suffix.lower()
    if ext == ".gif":
        imageio.mimsave(str(out_path), frames, fps=args.fps, loop=0)
    else:
        imageio.mimsave(str(out_path), frames, fps=args.fps,
                        codec="libx264", quality=8)

    print(f"Done → {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
