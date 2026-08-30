"""
Inverse Kinematics (IK) Controller for Agilex Piper 6-DOF Robotic Arm.
----------------------------------------------------------------------
Supports Pinocchio analytical Lie group kinematics as primary engine with
MuJoCo native forward kinematics & analytical Jacobians as fallback.
"""

from typing import Optional, Tuple
import numpy as np
import mujoco

PINOCCHIO_AVAILABLE = False


def mat2quat(rot_mat: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a unit quaternion [w, x, y, z]."""
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, rot_mat.flatten())
    return quat


def quat2mat(quat: np.ndarray) -> np.ndarray:
    """Convert a quaternion [w, x, y, z] to a 3x3 rotation matrix."""
    rot_mat = np.zeros(9)
    mujoco.mju_quat2Mat(rot_mat, quat)
    return rot_mat.reshape(3, 3)


def euler2mat(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Extrinsic XYZ roll-pitch-yaw to 3x3 rotation matrix."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def mat2euler(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert 3x3 rotation matrix to roll, pitch, yaw (XYZ extrinsic)."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0.0
    return float(x), float(y), float(z)


class DifferentialIKController:
    """
    High-precision 6-DOF Kinematics & Inverse Kinematics Controller.
    Uses Pinocchio Lie group kinematics when available, with MuJoCo C-kinematics fallback.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "ee",
        num_arm_joints: int = 6,
        max_iters: int = 50,
        tol_pos: float = 1e-4,     # 0.1 mm
        tol_rot: float = 1e-3,     # ~0.05 deg
        home_qpos: Optional[np.ndarray] = None,
        use_pinocchio: bool = False,
    ):
        self.model = model
        self.site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if self.site_id < 0:
            raise ValueError(f"Site '{site_name}' not found in MuJoCo model.")

        self.num_arm_joints = num_arm_joints
        self.max_iters = max_iters
        self.tol_pos = tol_pos
        self.tol_rot = tol_rot

        self.joint_limits_lower = self.model.jnt_range[:num_arm_joints, 0].copy()
        self.joint_limits_upper = self.model.jnt_range[:num_arm_joints, 1].copy()

        if home_qpos is not None:
            self.home_qpos = np.array(home_qpos[:num_arm_joints], dtype=np.float64)
        else:
            self.home_qpos = np.array([0.0, 1.5, -1.2, 0.0, 0.0, 0.0])

        # Initialize Pinocchio backend if requested & available
        self.pin_ik: Optional[PinocchioIKController] = None
        if use_pinocchio and PINOCCHIO_AVAILABLE:
            try:
                self.pin_ik = PinocchioIKController(
                    num_arm_joints=num_arm_joints,
                    max_iters=max_iters,
                    eps=tol_pos,
                    home_qpos=self.home_qpos,
                )
            except Exception:
                self.pin_ik = None

        # Scratchpad data for MuJoCo fallback
        self._d_scratch = mujoco.MjData(self.model)
        self._jacp = np.zeros((3, self.model.nv))
        self._jacr = np.zeros((3, self.model.nv))

        # Current setpoint state
        self.q_target = self.home_qpos.copy()
        self.target_pos = np.zeros(3)
        self.target_rot = np.eye(3)

    def reset(self, qpos: np.ndarray) -> None:
        """Reset internal setpoints to given joint configuration."""
        self.q_target = np.array(qpos[:self.num_arm_joints], dtype=np.float64)
        if self.pin_ik is not None:
            self.pin_ik.reset(self.q_target)

        self._d_scratch.qpos[:self.num_arm_joints] = self.q_target
        mujoco.mj_fwdPosition(self.model, self._d_scratch)
        self.target_pos = self._d_scratch.site_xpos[self.site_id].copy()
        self.target_rot = self._d_scratch.site_xmat[self.site_id].reshape(3, 3).copy()

    def get_current_ee_pose(self, data: mujoco.MjData) -> Tuple[np.ndarray, np.ndarray]:
        """Return (position [3], rotation_matrix [3,3]) of TCP in world frame."""
        pos = data.site_xpos[self.site_id].copy()
        rot = data.site_xmat[self.site_id].reshape(3, 3).copy()
        return pos, rot

    def solve_lm(
        self,
        target_pos: np.ndarray,
        target_rot: Optional[np.ndarray] = None,
        q_seed: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solve exact Inverse Kinematics using iterative Levenberg-Marquardt algorithm.
        """
        q = self.q_target.copy() if q_seed is None else q_seed.copy()
        damping = 1e-3
        track_rot = target_rot is not None

        q_min = self.joint_limits_lower
        q_max = self.joint_limits_upper
        q_mid = (q_max + q_min) / 2.0
        q_span = (q_max - q_min) / 2.0

        for it in range(self.max_iters):
            self._d_scratch.qpos[:self.num_arm_joints] = q
            mujoco.mj_fwdPosition(self.model, self._d_scratch)

            curr_p = self._d_scratch.site_xpos[self.site_id]
            curr_r = self._d_scratch.site_xmat[self.site_id].reshape(3, 3)

            e_pos = target_pos - curr_p

            if track_rot:
                R_err = target_rot @ curr_r.T
                trace = np.trace(R_err)
                cos_t = np.clip((trace - 1.0) / 2.0, -1.0, 1.0)
                theta = np.arccos(cos_t)
                if theta > 1e-5:
                    axis = np.array([
                        R_err[2, 1] - R_err[1, 2],
                        R_err[0, 2] - R_err[2, 0],
                        R_err[1, 0] - R_err[0, 1],
                    ]) / (2.0 * np.sin(theta))
                    e_rot = axis * theta
                else:
                    e_rot = np.zeros(3)
                err = np.concatenate([e_pos, 0.7 * e_rot])
                p_err = np.linalg.norm(e_pos)
                r_err = np.linalg.norm(e_rot)
                if p_err < self.tol_pos and r_err < self.tol_rot:
                    return q, True
            else:
                err = e_pos
                p_err = np.linalg.norm(e_pos)
                if p_err < self.tol_pos:
                    return q, True

            mujoco.mj_jacSite(self.model, self._d_scratch, self._jacp, self._jacr, self.site_id)
            if track_rot:
                J = np.vstack([self._jacp[:, :self.num_arm_joints], 0.7 * self._jacr[:, :self.num_arm_joints]])
                dim = 6
            else:
                J = self._jacp[:, :self.num_arm_joints]
                dim = 3

            # Smooth joint limit barrier weighting
            dist_norm = np.abs((q - q_mid) / (q_span + 1e-6))
            weights = 1.0 + 4.0 * (dist_norm ** 2)
            W_inv = np.diag(1.0 / weights)

            # Levenberg-Marquardt step
            JW = J @ W_inv
            A = JW @ J.T + damping * np.eye(dim)
            dq = W_inv @ J.T @ np.linalg.solve(A, err)

            # Nullspace posture regularization towards home pose
            null_proj = np.eye(self.num_arm_joints) - W_inv @ J.T @ np.linalg.solve(A, J)
            dq_null = 0.02 * null_proj @ (self.home_qpos - q)

            step_limit = 0.4
            max_dq = np.max(np.abs(dq))
            scale = min(1.0, step_limit / (max_dq + 1e-6))
            q_candidate = np.clip(q + scale * (dq + dq_null), q_min + 1e-4, q_max - 1e-4)

            self._d_scratch.qpos[:self.num_arm_joints] = q_candidate
            mujoco.mj_fwdPosition(self.model, self._d_scratch)
            new_p_err = np.linalg.norm(target_pos - self._d_scratch.site_xpos[self.site_id])

            if new_p_err < p_err:
                q = q_candidate
                damping = max(damping * 0.7, 1e-5)
            else:
                damping = min(damping * 1.5, 0.5)
                q = np.clip(q + 0.7 * scale * dq, q_min + 1e-4, q_max - 1e-4)

        return q, False

    def step_tcp_delta(
        self,
        delta_tcp_pos: np.ndarray,
        delta_tcp_rot: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Apply Cartesian step in the LOCAL TCP coordinate frame.
        """
        if self.pin_ik is not None:
            q_sol = self.pin_ik.step_tcp_delta(delta_tcp_pos, delta_tcp_rot)
            self.q_target = q_sol
            self._d_scratch.qpos[:self.num_arm_joints] = q_sol
            mujoco.mj_fwdPosition(self.model, self._d_scratch)
            self.target_pos = self._d_scratch.site_xpos[self.site_id].copy()
            self.target_rot = self._d_scratch.site_xmat[self.site_id].reshape(3, 3).copy()
            return q_sol

        # Fallback to internal LM solver
        delta_world_pos = self.target_rot @ delta_tcp_pos

        if delta_tcp_rot is not None and np.linalg.norm(delta_tcp_rot) > 1e-6:
            angle = float(np.linalg.norm(delta_tcp_rot))
            axis_local = delta_tcp_rot / angle
            delta_rot_mat = np.zeros(9)
            mujoco.mju_quat2Mat(delta_rot_mat, np.array([
                np.cos(angle / 2.0),
                axis_local[0] * np.sin(angle / 2.0),
                axis_local[1] * np.sin(angle / 2.0),
                axis_local[2] * np.sin(angle / 2.0),
            ]))
            new_target_rot = self.target_rot @ delta_rot_mat.reshape(3, 3)
        else:
            new_target_rot = self.target_rot.copy()

        new_target_pos = self.target_pos + delta_world_pos
        q_sol, success = self.solve_lm(new_target_pos, new_target_rot, q_seed=self.q_target)

        self.target_pos = new_target_pos
        self.target_rot = new_target_rot
        self.q_target = q_sol
        return self.q_target.copy()

    def step_cartesian_delta(
        self,
        data: mujoco.MjData,
        delta_pos: np.ndarray,
        delta_rot_axis_angle: Optional[np.ndarray] = None,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Apply Cartesian step in WORLD coordinates."""
        if self.pin_ik is not None:
            q_sol = self.pin_ik.step_world_delta(delta_pos, delta_rot_axis_angle)
            self.q_target = q_sol
            self._d_scratch.qpos[:self.num_arm_joints] = q_sol
            mujoco.mj_fwdPosition(self.model, self._d_scratch)
            self.target_pos = self._d_scratch.site_xpos[self.site_id].copy()
            self.target_rot = self._d_scratch.site_xmat[self.site_id].reshape(3, 3).copy()
            return q_sol

        new_target_pos = self.target_pos + delta_pos

        if delta_rot_axis_angle is not None and np.linalg.norm(delta_rot_axis_angle) > 1e-6:
            angle = float(np.linalg.norm(delta_rot_axis_angle))
            axis = delta_rot_axis_angle / angle
            delta_rot_mat = np.zeros(9)
            mujoco.mju_quat2Mat(delta_rot_mat, np.array([
                np.cos(angle / 2.0),
                axis[0] * np.sin(angle / 2.0),
                axis[1] * np.sin(angle / 2.0),
                axis[2] * np.sin(angle / 2.0),
            ]))
            new_target_rot = delta_rot_mat.reshape(3, 3) @ self.target_rot
        else:
            new_target_rot = self.target_rot.copy()

        q_sol, success = self.solve_lm(new_target_pos, new_target_rot, q_seed=self.q_target)

        self._d_scratch.qpos[:self.num_arm_joints] = q_sol
        mujoco.mj_fwdPosition(self.model, self._d_scratch)
        self.target_pos = self._d_scratch.site_xpos[self.site_id].copy()
        self.target_rot = self._d_scratch.site_xmat[self.site_id].reshape(3, 3).copy()
        self.q_target = q_sol
        return self.q_target.copy()

    def step_decoupled_delta(
        self,
        delta_workspace_pos: np.ndarray,
        delta_tool_rpy: np.ndarray,
    ) -> np.ndarray:
        """
        Decoupled Cartesian IK step with persistent target tracking:
          - delta_workspace_pos : [dX, dY, dZ] in World/Workspace Frame
          - delta_tool_rpy       : [dRoll, dPitch, dYaw] in Local Tool/Gripper Frame
        """
        # World Translation
        new_target_pos = self.target_pos + delta_workspace_pos

        # Local Tool Rotation
        if np.any(np.abs(delta_tool_rpy) > 1e-6):
            R_delta_local = euler2mat(delta_tool_rpy[0], delta_tool_rpy[1], delta_tool_rpy[2])
            new_target_rot = self.target_rot @ R_delta_local
        else:
            new_target_rot = self.target_rot.copy()

        q_sol, success = self.solve_lm(new_target_pos, new_target_rot, q_seed=self.q_target)

        self._d_scratch.qpos[:self.num_arm_joints] = q_sol
        mujoco.mj_fwdPosition(self.model, self._d_scratch)
        self.target_pos = self._d_scratch.site_xpos[self.site_id].copy()
        self.target_rot = self._d_scratch.site_xmat[self.site_id].reshape(3, 3).copy()
        self.q_target = q_sol
        return self.q_target.copy()

    def solve_decoupled(
        self,
        *args,
        **kwargs,
    ) -> np.ndarray:
        """
        Supports both signatures:
          1. solve_decoupled(delta_workspace_pos, delta_tool_rpy)
          2. solve_decoupled(qpos, ee_pos, ee_mat, delta_workspace_pos, delta_tool_rpy)
        """
        if len(args) == 2:
            return self.step_decoupled_delta(args[0], args[1])
        elif len(args) >= 5:
            return self.step_decoupled_delta(args[3], args[4])
        elif "delta_workspace_pos" in kwargs and "delta_tool_rpy" in kwargs:
            return self.step_decoupled_delta(kwargs["delta_workspace_pos"], kwargs["delta_tool_rpy"])
        else:
            raise ValueError(f"Invalid arguments to solve_decoupled: args={args}, kwargs={kwargs}")

    def solve(
        self,
        data: mujoco.MjData,
        target_pos: Optional[np.ndarray] = None,
        target_rot: Optional[np.ndarray] = None,
        dt: float = 0.02,
    ) -> np.ndarray:
        """Track arbitrary Cartesian target pose."""
        if target_pos is None:
            target_pos = self.target_pos
        else:
            self.target_pos = target_pos.copy()

        if target_rot is not None:
            self.target_rot = target_rot.copy()

        q_sol, _ = self.solve_lm(self.target_pos, self.target_rot if target_rot is not None else None, q_seed=self.q_target)
        self.q_target = q_sol
        return self.q_target.copy()

