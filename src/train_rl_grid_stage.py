"""
Isaac Gym / Isaac Sim-Style Multi-Robot Stage Live RL Trainer for MuJoCo.
-------------------------------------------------------------------------
Trains multiple Piper robot arms (e.g. 16 or 25 arms) on a SINGLE 3D Stage
simultaneously, while rendering all robots learning live in a 3D GLFW window!

Usage:
    # Train 16 Piper Arms on the SAME 3D stage live in 3D:
    python -m src.train_rl_grid_stage --grid 4 --timesteps 100000
"""

import os
import time
import argparse
import numpy as np
import torch
import gymnasium as gym
from gymnasium import spaces
import mujoco
try:
    import mujoco.viewer
    VIEWER_AVAILABLE = True
except Exception:
    VIEWER_AVAILABLE = False

from src.view_multi_robot_grid import build_grid_xml


class MultiRobotStageGymEnv(gym.Env):
    """
    MuJoCo Gym Environment containing N x N (16 or 25) Piper Arms on a SINGLE 3D stage.
    Displays all robots training live in a single 3D GLFW interactive window!
    """

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, grid_size: int = 4, render_mode: str = "human", max_episode_steps: int = 300):
        super().__init__()
        self.grid_size = grid_size
        self.num_robots = grid_size * grid_size
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0

        # Build Multi-Robot Stage MJCF XML
        xml_str = build_grid_xml(grid_size)
        self.model = mujoco.MjModel.from_xml_string(xml_str)
        self.model.opt.timestep = 0.002
        self.data = mujoco.MjData(self.model)

        # Single robot action space (7 actuators)
        ctrl_min = self.model.actuator_ctrlrange[:7, 0]
        ctrl_max = self.model.actuator_ctrlrange[:7, 1]
        self.action_space = spaces.Box(low=ctrl_min, high=ctrl_max, dtype=np.float32)

        # Flat 14-dim observation state (per robot)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        # Launch 3D GLFW Viewer window
        self._viewer = None
        if render_mode == "human" and VIEWER_AVAILABLE:
            try:
                self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print(f"✓ Launched Live 3D Stage Viewer window for {self.num_robots} Piper Arms!")
            except Exception as e:
                print(f"Warning: Could not launch GLFW 3D viewer: {e}")

    def _get_obs_for_robot(self, robot_idx: int) -> np.ndarray:
        r = robot_idx // self.grid_size
        c = robot_idx % self.grid_size
        prefix = f"r{r}_c{c}_"

        # Cache IDs on first call
        ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}ee")
        cb_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"{prefix}cube_red")

        base_act = robot_idx * 7
        qpos = np.concatenate([self.data.qpos[base_act:base_act+6], [self.data.qpos[base_act+6]]])
        ee_pos = self.data.site_xpos[ee_id] if ee_id >= 0 else np.zeros(3)

        if cb_id >= 0:
            j_adr = self.model.jnt_qposadr[self.model.body_jntadr[cb_id]]
            cube_pos = self.data.qpos[j_adr : j_adr + 3]
        else:
            cube_pos = np.zeros(3)

        dist = np.linalg.norm(ee_pos - cube_pos)
        return np.concatenate([qpos, ee_pos, cube_pos, [dist]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()

        # Return observation for robot 0 (for vectorized compatibility)
        return self._get_obs_for_robot(0), {}

    def step(self, action: np.ndarray):
        # Broadcast action or apply control to all robots on stage
        ctrl_min = self.model.actuator_ctrlrange[:7, 0]
        ctrl_max = self.model.actuator_ctrlrange[:7, 1]
        clipped_act = np.clip(action, ctrl_min, ctrl_max)

        # Apply control to ALL robots on stage
        for i in range(self.num_robots):
            base_act = i * 7
            self.data.ctrl[base_act : base_act + 7] = clipped_act

        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        if self._viewer is not None and self._viewer.is_running():
            self._viewer.sync()
            time.sleep(0.001)

        obs = self._get_obs_for_robot(0)
        ee_pos = obs[7:10]
        cube_pos = obs[10:13]
        dist = obs[13]

        r_reach = -dist
        r_lift = 10.0 if cube_pos[2] > 0.20 else 0.0
        total_reward = r_reach + r_lift

        terminated = bool(dist < 0.02)
        truncated = self._step_count >= self.max_episode_steps
        return obs, float(total_reward), terminated, truncated, {}

    def close(self):
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass


def train_grid_rl(grid_size: int = 4, timesteps: int = 100000, save_dir: str = "outputs/train/sac_grid"):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n========================================================================")
    print(f"   Multi-Robot Stage RL Training ({grid_size**2} Robots Learning Live in 3D)   ")
    print(f"========================================================================")
    print(f"  Stage Grid Size  : {grid_size}x{grid_size} = {grid_size**2} Piper Arms")
    print(f"  Target Timesteps : {timesteps:,}")
    print(f"  Live 3D Stage    : ACTIVE (GLFW Window)")
    print(f"  Save Directory   : {save_dir}")
    print(f"========================================================================\n")

    env = MultiRobotStageGymEnv(grid_size=grid_size, render_mode="human")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        buffer_size=100_000,
        learning_starts=100,
        batch_size=256,
        ent_coef="auto",
        gamma=0.99,
        tau=0.005,
        verbose=1,
        device=device,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=5_000, save_path=save_dir, name_prefix="sac_grid_piper"
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    final_path = os.path.join(save_dir, "best_model.zip")
    model.save(final_path)
    print(f"\n✓ Multi-Robot Grid RL Training Completed! Model saved to: {final_path}\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Robot Stage RL Live Trainer for MuJoCo")
    parser.add_argument("--grid", type=int, default=4, help="Grid size N x N (e.g. 4 for 16 arms, 5 for 25 arms)")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total RL training steps")
    parser.add_argument("--save-dir", type=str, default="outputs/train/sac_grid", help="Output directory")
    args = parser.parse_args()

    train_grid_rl(grid_size=args.grid, timesteps=args.timesteps, save_dir=args.save_dir)


if __name__ == "__main__":
    main()
