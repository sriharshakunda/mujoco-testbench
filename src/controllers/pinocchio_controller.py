"""
Pinocchio-based Kinematics & Inverse Kinematics Controller for Agilex Piper Arm.
--------------------------------------------------------------------------------
Uses Pinocchio's analytical Lie Group SE(3) kinematics, analytical Jacobians,
and Newton-Raphson / Damped Least Squares (DLS) closed-loop IK.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple, Any
import numpy as np

try:
    import pinocchio as pin
    PINOCCHIO_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    pin = None
    PINOCCHIO_AVAILABLE = False


class PinocchioIKController:
    """
    6-DOF Task-Space Kinematics & IK Controller powered by Pinocchio.
    """

    def __init__(
        self,
        urdf_path: Optional[str] = None,
        tcp_frame_name: str = "gripper_tcp",
        num_arm_joints: int = 6,
        max_iters: int = 30,
        eps: float = 1e-4,
        damping: float = 1e-4,
        home_qpos: Optional[np.ndarray] = None,
    ):
        if not PINOCCHIO_AVAILABLE or pin is None:
            raise ImportError(
                "Pinocchio is not installed in this environment. "
                "Please install pinocchio (`pip install pin` or `pip install pinocchio`)."
            )

        if urdf_path is None:
            urdf_path = str(Path(__file__).parents[2] / "assets" / "piper" / "piper_with_gripper.urdf")

        with open(urdf_path, "r") as f:
            urdf_content = f.read()

        # Build Pinocchio model from URDF
        self.model = pin.buildModelFromXML(urdf_content)
        self.data = self.model.createData()

        # Find TCP frame ID
        if self.model.existFrame(tcp_frame_name):
            self.tcp_frame_id = self.model.getFrameId(tcp_frame_name)
        elif self.model.existFrame("gripper_base"):
            self.tcp_frame_id = self.model.getFrameId("gripper_base")
        else:
            self.tcp_frame_id = self.model.getFrameId("link6")

        self.num_arm_joints = num_arm_joints
        self.max_iters = max_iters
        self.eps = eps
        self.damping = damping

        self.q_min = self.model.lowerPositionLimit[:num_arm_joints].copy()
        self.q_max = self.model.upperPositionLimit[:num_arm_joints].copy()

        if home_qpos is not None:
            self.home_qpos = np.array(home_qpos[:num_arm_joints], dtype=np.float64)
        else:
            self.home_qpos = np.zeros(num_arm_joints, dtype=np.float64)
        self.q_current = np.zeros(self.model.nq)
        self.q_current[:num_arm_joints] = self.home_qpos
        self.q_target = self.home_qpos.copy()

        # Target SE(3) pose
        self.target_SE3 = pin.SE3.Identity()
        self.reset(self.home_qpos)

    def reset(self, qpos: np.ndarray) -> None:
        """Reset internal state to given joint angles."""
        self.q_current = np.zeros(self.model.nq)
        self.q_current[:self.num_arm_joints] = qpos[:self.num_arm_joints]
        self.q_target = self.q_current[:self.num_arm_joints].copy()

        pin.forwardKinematics(self.model, self.data, self.q_current)
        pin.updateFramePlacements(self.model, self.data)
        self.target_SE3 = self.data.oMf[self.tcp_frame_id].copy()

    def get_current_ee_pose(self, qpos: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return (translation [3], rotation [3,3]) of TCP."""
        q = np.zeros(self.model.nq)
        q[:self.num_arm_joints] = qpos[:self.num_arm_joints]
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        oMf = self.data.oMf[self.tcp_frame_id]
        return oMf.translation.copy(), oMf.rotation.copy()

    def solve_ik(
        self,
        target_SE3: Any,
        q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve exact Inverse Kinematics for a desired SE(3) pose using Pinocchio.
        """
        q = self.q_current.copy()
        if q_seed is not None:
            q[:self.num_arm_joints] = q_seed[:self.num_arm_joints]

        q_mid = (self.q_max + self.q_min) / 2.0
        q_span = (self.q_max - self.q_min) / 2.0

        for it in range(self.max_iters):
            pin.forwardKinematics(self.model, self.data, q)
            pin.updateFramePlacements(self.model, self.data)
            oMf_curr = self.data.oMf[self.tcp_frame_id]

            # Compute error in local TCP frame via Lie group log6
            dMf = oMf_curr.inverse() * target_SE3
            err = pin.log6(dMf).vector

            err_norm = np.linalg.norm(err)
            if err_norm < self.eps:
                return q[:self.num_arm_joints].copy(), True

            # Compute analytical spatial Jacobian in LOCAL frame
            J = pin.computeFrameJacobian(
                self.model, self.data, q, self.tcp_frame_id, pin.LOCAL
            )[:, :self.num_arm_joints]

            # Joint limit barrier weighting
            dist_norm = np.abs((q[:self.num_arm_joints] - q_mid) / (q_span + 1e-6))
            weights = 1.0 + 10.0 * (dist_norm ** 4)
            W_inv = np.diag(1.0 / weights)

            # Damped Least Squares
            JW = J @ W_inv
            A = JW @ J.T + (self.damping ** 2) * np.eye(6)
            dq = W_inv @ J.T @ np.linalg.solve(A, err)

            # Nullspace posture bias
            null_proj = np.eye(self.num_arm_joints) - W_inv @ J.T @ np.linalg.solve(A, J)
            dq_null = 0.05 * null_proj @ (self.home_qpos - q[:self.num_arm_joints])

            step_limit = 0.3
            max_dq = np.max(np.abs(dq))
            scale = min(1.0, step_limit / (max_dq + 1e-6))
            dq_total = scale * (dq + dq_null)

            q[:self.num_arm_joints] = np.clip(
                q[:self.num_arm_joints] + dq_total,
                self.q_min + 1e-4,
                self.q_max - 1e-4,
            )

        return q[:self.num_arm_joints].copy(), False

    def step_tcp_delta(
        self,
        delta_tcp_pos: np.ndarray,
        delta_tcp_rot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Step in the TCP local coordinate frame:
        - delta_tcp_pos[2]: approach axis (forward/back)
        - delta_tcp_pos[0]: lateral axis (left/right)
        - delta_tcp_pos[1]: vertical/slide axis (up/down)
        """
        d_trans = delta_tcp_pos

        if delta_tcp_rot is not None and np.linalg.norm(delta_tcp_rot) > 1e-6:
            angle = float(np.linalg.norm(delta_tcp_rot))
            axis = delta_tcp_rot / angle
            d_rot = pin.exp3(axis * angle)
        else:
            d_rot = np.eye(3)

        d_SE3 = pin.SE3(d_rot, d_trans)
        new_target_SE3 = self.target_SE3 * d_SE3

        q_sol, success = self.solve_ik(new_target_SE3, q_seed=self.q_target)
        self.target_SE3 = new_target_SE3
        self.q_target = q_sol
        self.q_current[:self.num_arm_joints] = q_sol
        return self.q_target.copy()

    def step_world_delta(
        self,
        delta_world_pos: np.ndarray,
        delta_world_rot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Step in the global WORLD coordinate frame."""
        new_translation = self.target_SE3.translation + delta_world_pos

        if delta_world_rot is not None and np.linalg.norm(delta_world_rot) > 1e-6:
            angle = float(np.linalg.norm(delta_world_rot))
            axis = delta_world_rot / angle
            R_inc = pin.exp3(axis * angle)
            new_rotation = R_inc @ self.target_SE3.rotation
        else:
            new_rotation = self.target_SE3.rotation.copy()

        new_target_SE3 = pin.SE3(new_rotation, new_translation)
        q_sol, success = self.solve_ik(new_target_SE3, q_seed=self.q_target)
        self.target_SE3 = new_target_SE3
        self.q_target = q_sol
        self.q_current[:self.num_arm_joints] = q_sol
        return self.q_target.copy()
