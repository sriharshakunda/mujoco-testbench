"""MuJoCo environment wrapper for Agilex Piper arm."""

import os
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
import mujoco

logger = logging.getLogger(__name__)

MJCF_PATH = Path(__file__).parents[2] / "assets" / "piper.xml"
N_ARM_JOINTS = 6
N_GRIPPER = 1        # single actuator; equality mirrors to finger2
N_JOINTS = N_ARM_JOINTS + N_GRIPPER

# Official ready configuration: arm upright, hovering over table workspace
HOME_QPOS = np.array([0.0, -3.10, -0.25, 0.0, 0.0, 0.0, 0.04, -0.04])
#                    j1   j2    j3     j4   j5    j6   g1    g2
HOME_CTRL = np.array([0.0, -3.10, -0.25, 0.0, 0.0, 0.0, 0.04])
#                    act1 act2  act3   act4 act5  act6 gripper
# Standard zero home configuration

class PiperEnv:
    """MuJoCo simulation of the Agilex Piper 6-DOF arm."""

    def __init__(
        self,
        render_mode: Optional[str] = None,  # "human" | "rgb_array" | None
        dt: float = 0.002,
        max_episode_steps: int = 500,
        target_pos: Optional[np.ndarray] = None,
    ):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        self._viewer = None

        self.model = mujoco.MjModel.from_xml_path(str(MJCF_PATH))
        self.model.opt.timestep = dt
        self.data = mujoco.MjData(self.model)

        # Cache IDs
        self._ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self._target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "target")

        if target_pos is not None:
            self.set_target(target_pos)

        logger.info("PiperEnv ready  (model=%s)", MJCF_PATH.name)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def randomize_cubes(self, rng: Optional[np.random.Generator] = None) -> None:
        """Randomize positions (X/Y on tabletop outside bins) and yaw orientation of pickable cubes."""
        base_positions = [
            np.array([0.28, 0.15, 0.165]),
            np.array([0.34, 0.15, 0.165]),
            np.array([0.31, 0.21, 0.165]),
        ]
        np_rng = np.random.default_rng() if rng is None else rng
        for i, cube_name in enumerate(["cube_red", "cube_yellow", "cube_purple"]):
            b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
            if b_id >= 0:
                j_adr = self.model.jnt_qposadr[self.model.body_jntadr[b_id]]
                # Jitter within safe table reach area
                d_xy = np_rng.uniform([-0.018, -0.018], [0.018, 0.018])
                pos = base_positions[i].copy()
                pos[0] += d_xy[0]
                pos[1] += d_xy[1]
                # Random yaw orientation [-pi, +pi]
                yaw = np_rng.uniform(-np.pi, np.pi)
                quat = np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])
                self.data.qpos[j_adr : j_adr + 7] = np.concatenate([pos, quat])

    def reset(self, randomize_cubes: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
        mujoco.mj_resetData(self.model, self.data)
        # Apply safe home pose immediately so links never intersect the table.
        self.data.qpos[:len(HOME_QPOS)] = HOME_QPOS
        self.data.ctrl[:len(HOME_CTRL)] = HOME_CTRL
        if randomize_cubes:
            self.randomize_cubes()
        mujoco.mj_forward(self.model, self.data)
        self._step_count = 0
        return self._get_obs(), {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Apply joint positions/torques and advance one timestep."""
        ctrl_min = self.model.actuator_ctrlrange[:, 0]
        ctrl_max = self.model.actuator_ctrlrange[:, 1]
        self.data.ctrl[:len(action)] = np.clip(action, ctrl_min[:len(action)], ctrl_max[:len(action)])
        mujoco.mj_step(self.model, self.data)
        self._step_count += 1

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = bool(reward > -0.02)       # within 2 cm of target
        truncated = self._step_count >= self.max_episode_steps
        return obs, reward, terminated, truncated, {}

    def set_target(self, pos: np.ndarray) -> None:
        """Move the green goal marker."""
        self.model.site_pos[self._target_id] = pos

    def get_ee_pos(self) -> np.ndarray:
        """Return world-frame end-effector position."""
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.site_xpos[self._ee_id].copy()

    def get_ee_mat(self) -> np.ndarray:
        """Return world-frame 3x3 end-effector rotation matrix."""
        mujoco.mj_fwdPosition(self.model, self.data)
        return self.data.site_xmat[self._ee_id].reshape(3, 3).copy()

    def get_target_pos(self) -> np.ndarray:
        return self.model.site_pos[self._target_id].copy()

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            renderer = mujoco.Renderer(self.model, height=480, width=640)
            renderer.update_scene(self.data)
            pixels = renderer.render()
            renderer.close()
            return pixels
        return None

    def close(self) -> None:
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception:
                pass
            self._viewer = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def joint_pos(self) -> np.ndarray:
        return self.data.qpos[:N_JOINTS].copy()

    @property
    def joint_vel(self) -> np.ndarray:
        return self.data.qvel[:N_JOINTS].copy()

    @property
    def observation_space_shape(self) -> Tuple[int, ...]:
        # qpos(7) + qvel(7) + ee_pos(3) + target_pos(3)
        return (N_JOINTS * 2 + 6,)

    @property
    def action_space_shape(self) -> Tuple[int, ...]:
        return (N_JOINTS,)  # 6 arm + 1 gripper

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        mujoco.mj_fwdPosition(self.model, self.data)
        ee_pos = self.data.site_xpos[self._ee_id]
        target_pos = self.model.site_pos[self._target_id]
        return np.concatenate([
            self.data.qpos[:N_JOINTS],
            self.data.qvel[:N_JOINTS],
            ee_pos,
            target_pos,
        ])

    def _compute_reward(self, obs: np.ndarray) -> float:
        ee_pos = obs[N_JOINTS * 2: N_JOINTS * 2 + 3]
        target_pos = obs[N_JOINTS * 2 + 3: N_JOINTS * 2 + 6]
        dist = float(np.linalg.norm(ee_pos - target_pos))
        return -dist

    # Context manager support
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


# ------------------------------------------------------------------
# Gymnasium Interface Integration
# ------------------------------------------------------------------
try:
    import gymnasium as gym
    from gymnasium import spaces
    GYM_AVAILABLE = True
except ImportError:
    try:
        import gym
        from gym import spaces
        GYM_AVAILABLE = True
    except ImportError:
        GYM_AVAILABLE = False

if GYM_AVAILABLE:
    class PiperGymEnv(gym.Env):
        """Gymnasium environment wrapper for Agilex Piper MuJoCo simulation."""
        metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

        def __init__(self, render_mode: Optional[str] = None, max_episode_steps: int = 300):
            super().__init__()
            self.env = PiperEnv(render_mode=render_mode, max_episode_steps=max_episode_steps)
            self.render_mode = render_mode
            self.max_episode_steps = max_episode_steps
            self._max_episode_steps = max_episode_steps
            self.task = "pick up the red block and place in blue bin"
            self.task_description = "pick up the red block and place in blue bin"

            from src.camera import WristCamera
            self.wrist_cam = WristCamera(self.env.model, "wrist_rgb", height=480, width=640)
            self.scene_cam = WristCamera(self.env.model, "scene_cam", height=480, width=640)

            ctrl_min = self.env.model.actuator_ctrlrange[:, 0]
            ctrl_max = self.env.model.actuator_ctrlrange[:, 1]
            self.action_space = spaces.Box(low=ctrl_min, high=ctrl_max, dtype=np.float32)

            self.observation_space = spaces.Dict({
                "observation.state": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "observation.images.wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "observation.images.extrinsic": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                "agent_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
                "pixels": spaces.Dict({
                    "wrist": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                    "extrinsic": spaces.Box(low=0, high=255, shape=(480, 640, 3), dtype=np.uint8),
                }),
            })

        def _get_obs(self) -> dict:
            state = np.concatenate([self.env.data.qpos[:6], [self.env.data.qpos[6]]]).astype(np.float32)
            w_rgb = self.wrist_cam.get_rgb(self.env.data)
            s_rgb = self.scene_cam.get_rgb(self.env.data)
            return {
                "observation.state": state,
                "observation.images.wrist": w_rgb,
                "observation.images.extrinsic": s_rgb,
                "agent_pos": state,
                "pixels": {
                    "wrist": w_rgb,
                    "extrinsic": s_rgb,
                }
            }

        def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
            if seed is not None:
                np.random.seed(seed)
            self.env.reset(randomize_cubes=True)
            obs_dict = self._get_obs()
            return obs_dict, {}

        def step(self, action: np.ndarray):
            _, reward, terminated, truncated, info = self.env.step(action)
            obs_dict = self._get_obs()
            return obs_dict, reward, terminated, truncated, info

        def render(self):
            return self.env.render()

        def close(self):
            self.env.close()

def make_env(n_envs: int = 1, use_async_envs: bool = False, render_mode: Optional[str] = None):
    """
    Create vectorized environments for MuJoCo Piper task (LeRobot EnvHub Standard).
    """
    if not GYM_AVAILABLE:
        raise ImportError("Gymnasium is required to create vectorized environments.")

    # Prevent X11 display collisions in parallel worker processes
    if "MUJOCO_GL" not in os.environ:
        if use_async_envs or n_envs > 1 or render_mode != "human":
            os.environ["MUJOCO_GL"] = "egl"

    def _make_single_env():
        return PiperGymEnv(render_mode=render_mode)

    env_cls = gym.vector.AsyncVectorEnv if use_async_envs else gym.vector.SyncVectorEnv
    return env_cls([_make_single_env for _ in range(n_envs)])

