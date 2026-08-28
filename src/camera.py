"""
Wrist camera module for the Piper arm.

Provides RGB images, metric depth images, and 3-D point clouds from
the cameras defined in piper.xml (wrist_rgb / wrist_depth).

Coordinate conventions
----------------------
- MuJoCo camera frame : X right, Y up, Z backward (optical axis = -Z).
- Depth returned by MuJoCo : values in [0, 1] (normalised depth buffer).
  We linearise these to actual metres using the near/far clip planes.
- Point cloud : returned in world frame (metres).
"""

import os
import numpy as np
import mujoco

if "MUJOCO_GL" not in os.environ and "DISPLAY" not in os.environ:
    os.environ["MUJOCO_GL"] = "egl"


class WristCamera:
    """Renders RGB and depth from a named MuJoCo camera.

    Parameters
    ----------
    model     : mujoco.MjModel
    cam_name  : camera name as declared in the MJCF (default "wrist_rgb")
    height    : render resolution in pixels
    width     : render resolution in pixels
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cam_name: str = "wrist_rgb",
        height: int = 240,
        width: int = 320,
        exposure: float = 1.0,
    ):
        self.model    = model
        self.height   = height
        self.width    = width
        self.exposure = exposure
        self.cam_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if self.cam_id < 0:
            raise ValueError(f"Camera '{cam_name}' not found in model.")

        self._renderer = mujoco.Renderer(model, height, width)

        # Camera intrinsics (pinhole, square pixels)
        fovy_rad     = np.deg2rad(model.cam_fovy[self.cam_id])
        self.fy      = (height / 2.0) / np.tan(fovy_rad / 2.0)
        self.fx      = self.fy                  # square pixels
        self.cx      = width  / 2.0
        self.cy      = height / 2.0

    # ------------------------------------------------------------------
    # 2-D outputs
    # ------------------------------------------------------------------

    def get_rgb(self, data: mujoco.MjData) -> np.ndarray:
        """Return H×W×3 uint8 RGB image."""
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_id)
        rgb = self._renderer.render().copy()
        if self.exposure != 1.0:
            rgb = np.clip(rgb.astype(np.float32) * self.exposure, 0, 255).astype(np.uint8)
        return rgb

    def get_depth(self, data: mujoco.MjData) -> np.ndarray:
        """Return H×W float32 depth image in metres.

        MuJoCo 3.x renders depth directly in metres (no normalisation needed).
        """
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_id)
        depth_m = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()
        return depth_m

    def get_rgb_and_depth(self, data: mujoco.MjData):
        """Return (rgb, depth_m) in one call to minimise scene updates."""
        # RGB
        self._renderer.disable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_id)
        rgb = self._renderer.render().copy()
        if self.exposure != 1.0:
            rgb = np.clip(rgb.astype(np.float32) * self.exposure, 0, 255).astype(np.uint8)

        # Depth — MuJoCo 3.x returns metres directly
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(data, camera=self.cam_id)
        depth_m = self._renderer.render().copy()
        self._renderer.disable_depth_rendering()

        return rgb, depth_m

    # ------------------------------------------------------------------
    # 3-D output
    # ------------------------------------------------------------------

    def get_pointcloud(
        self,
        data: mujoco.MjData,
        max_depth: float = 2.0,
    ) -> np.ndarray:
        """Return Nx3 float32 point cloud in world coordinates (metres).

        Parameters
        ----------
        data      : live simulation data
        max_depth : discard points farther than this (metres)
        """
        depth_m = self.get_depth(data)

        # --- back-project to camera frame ---
        u = np.arange(self.width,  dtype=np.float32)
        v = np.arange(self.height, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)                   # H×W each

        # MuJoCo camera looks in -Z; depth is distance along optical axis
        x_cam =  (uu - self.cx) * depth_m / self.fx
        y_cam = -(vv - self.cy) * depth_m / self.fy  # flip Y (image Y down, cam Y up)
        z_cam = -depth_m                              # optical axis is -Z

        pts_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)  # H×W×3

        # --- transform to world frame ---
        cam_pos = data.cam_xpos[self.cam_id]          # (3,)
        cam_rot = data.cam_xmat[self.cam_id].reshape(3, 3)   # row-major rotation

        pts_flat  = pts_cam.reshape(-1, 3)            # N×3
        pts_world = (cam_rot @ pts_flat.T).T + cam_pos

        # --- filter by depth ---
        mask = depth_m.reshape(-1) < max_depth
        return pts_world[mask].astype(np.float32)

    def close(self) -> None:
        self._renderer.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
