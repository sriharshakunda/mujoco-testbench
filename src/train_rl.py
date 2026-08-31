"""
State-Based Soft Actor-Critic (SAC) Reinforcement Learning Trainer for MuJoCo Piper Arm.
---------------------------------------------------------------------------------------
Trains a continuous RL policy from scratch directly inside the MuJoCo simulation environment
WITHOUT requiring any pre-collected human demonstrations or foundation datasets.

Usage:
    # Train SAC RL policy for 100,000 steps locally (~10-15 minutes):
    python -m src.train_rl --timesteps 100000 --save-dir outputs/train/sac_piper

    # Evaluate trained SAC RL policy:
    python -m src.train_rl --eval --model-path outputs/train/sac_piper/best_model.zip
"""

import os
import time
import argparse
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import torch
import mujoco

from src.environment.env import PiperEnv, MJCF_PATH


class PiperRLFlatGymEnv(gym.Env):
    """
    Flat Vector State-Based Gym Environment tailored for fast RL algorithms (SAC/TD-MPC/PPO).
    Observation (14-dim):
        - Joint positions & gripper (7)
        - End-effector position XYZ (3)
        - Red Cube position XYZ (3)
        - EE-to-Cube Distance (1)
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, max_episode_steps=300):
        super().__init__()
        self.env = PiperEnv(render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps

        # Action space: 6 arm joint targets + 1 gripper target
        ctrl_min = self.env.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.env.model.actuator_ctrlrange[:, 1]
        self.action_space = spaces.Box(low=ctrl_min, high=ctrl_max, dtype=np.float32)

        # Flat 14-dim observation state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(14,), dtype=np.float32
        )

        self._cube_id = mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_BODY, "cube_red")

    def _get_obs(self) -> np.ndarray:
        qpos = np.concatenate([self.env.data.qpos[:6], [self.env.data.qpos[6]]])
        ee_pos = self.env.data.site_xpos[self.env._ee_id]

        if self._cube_id >= 0:
            j_adr = self.env.model.jnt_qposadr[self.env.model.body_jntadr[self._cube_id]]
            cube_pos = self.env.data.qpos[j_adr : j_adr + 3]
        else:
            cube_pos = np.zeros(3)

        dist = np.linalg.norm(ee_pos - cube_pos)
        return np.concatenate([qpos, ee_pos, cube_pos, [dist]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.env.reset(randomize_cubes=True)
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        obs_dict, reward, terminated, truncated, info = self.env.step(action)
        flat_obs = self._get_obs()

        ee_pos = flat_obs[7:10]
        cube_pos = flat_obs[10:13]
        gripper_pos = flat_obs[6]  # 0.0 = closed, 0.04 = open

        # -------------------------------------------------------------
        # Phase 1: Precision Parking Directly Above Red Cube (Hover 3cm)
        # -------------------------------------------------------------
        target_hover_pos = np.array([cube_pos[0], cube_pos[1], 0.20])  # 3.5cm above tabletop
        dist_to_hover = np.linalg.norm(ee_pos - target_hover_pos)

        # 1. Reaching Hover Target Penalty
        r_reach = -dist_to_hover

        # 2. Precision Parking Bonus (< 1.5 cm from hover target with open gripper)
        r_park = 0.0
        if dist_to_hover < 0.015:
            r_park = +10.0
            if gripper_pos > 0.03:  # Open gripper ready for grasp
                r_park += +20.0
                info["success"] = True
                if dist_to_hover < 0.008:  # Spot-on 8mm parking precision
                    r_park += +50.0

        total_reward = r_reach + r_park
        return flat_obs, float(total_reward), terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def train_sac(timesteps: int, save_dir: str):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback

    os.makedirs(save_dir, exist_ok=True)
    print(f"\n========================================================================")
    print(f"       Training Soft Actor-Critic (SAC) RL Policy from Scratch         ")
def train_sac(timesteps: int, save_dir: str, n_envs: int = 1, render: bool = False):
    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.vec_env import SubprocVecEnv

    os.makedirs(save_dir, exist_ok=True)
    render_mode = "human" if render else None
    print(f"\n========================================================================")
    print(f"       RL Training (MuJoCo Live 3D Viewer = {render})                  ")
    print(f"========================================================================")
    print(f"  Target Timesteps : {timesteps:,}")
    print(f"  Parallel Envs    : {n_envs} Robots")
    print(f"  Live 3D Window   : {render_mode}")
    print(f"  Save Directory   : {save_dir}")
    print(f"  Device           : cuda if available, else cpu")
    print(f"========================================================================\n")

    if not render:
        os.environ["MUJOCO_GL"] = "egl"

    def make_env():
        return PiperRLFlatGymEnv(render_mode=render_mode)

    if n_envs > 1:
        env = SubprocVecEnv([make_env for _ in range(n_envs)])
    else:
        env = PiperRLFlatGymEnv(render_mode=render_mode)

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
        save_freq=max(1000, 10_000 // n_envs), save_path=save_dir, name_prefix="sac_piper"
    )

    model.learn(total_timesteps=timesteps, callback=checkpoint_callback)
    final_path = os.path.join(save_dir, "best_model.zip")
    model.save(final_path)
    print(f"\n✓ RL Training Completed! Policy saved to: {final_path}\n")


def eval_sac(model_path: str, episodes: int = 10, target_fps: int = 30):
    from stable_baselines3 import SAC

    print(f"\nEvaluating SAC RL Policy from '{model_path}' ({episodes} Episodes at ~{target_fps} FPS)...")
    env = PiperRLFlatGymEnv(render_mode="human")
    model = SAC.load(model_path)
    dt = 1.0 / target_fps

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        step = 0
        total_r = 0.0

        while not done and step < 300:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_r += r
            step += 1
            time.sleep(dt)  # Smooth real-time 30 FPS playback

        status = "✓ SUCCESS!" if info.get("success", False) else "Finished"
        print(f"Episode {ep+1}/{episodes}: {status} | Total Reward = {total_r:.2f}, Steps = {step}")
    
    try:
        env.close()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="State-Based SAC RL Trainer for Piper Arm")
    parser.add_argument("--timesteps", type=int, default=100000, help="Total RL training steps")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel robots")
    parser.add_argument("--render", action="store_true", help="Open live interactive 3D GLFW viewer window during training")
    parser.add_argument("--save-dir", type=str, default="outputs/train/sac_piper", help="Output directory")
    parser.add_argument("--eval", action="store_true", help="Evaluate a trained SAC model")
    parser.add_argument("--model-path", type=str, default="outputs/train/sac_piper/best_model.zip", help="Path to SAC zip model")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS during evaluation (default: 30)")
    args = parser.parse_args()

    if args.eval:
        eval_sac(args.model_path, episodes=args.episodes, target_fps=args.fps)
    else:
        train_sac(args.timesteps, args.save_dir, n_envs=args.n_envs, render=args.render)


if __name__ == "__main__":
    main()
