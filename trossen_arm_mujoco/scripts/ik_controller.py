# Copyright 2025 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Trossen AI Robot IK Controller.

Provides inverse kinematics control using damped least squares differential IK
for WidowX AI, Stationary AI, and Mobile AI robots.
"""

from enum import Enum
from typing import Tuple

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation


class RobotType(Enum):
    """Robot type identifiers."""

    WXAI = "wxai"
    STATIONARY_AI = "stationary_ai"
    MOBILE_AI = "mobile_ai"


# Gripper position limits in meters
GRIPPER_OPEN_POSITION = 0.044
GRIPPER_CLOSED_POSITION = 0.022

# Default IK parameters
DEFAULT_IK_SCALE = 0.5
DEFAULT_IK_DAMPING = 0.03


class IKController:
    """IK controller for Trossen AI robots.

    Uses damped least squares differential IK to compute joint velocities
    that move the end effector towards target poses.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_type: RobotType = RobotType.WXAI,
        ee_site_name: str = "ee_site",
        arm_joint_names: list = None,
        gripper_joint_names: list = None,
        ik_scale: float = DEFAULT_IK_SCALE,
        ik_damping: float = DEFAULT_IK_DAMPING,
    ):
        """Initialize IK controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_type: Type of robot (WXAI, STATIONARY_AI, or MOBILE_AI)
            ee_site_name: Name of end effector site in the model
            arm_joint_names: List of arm joint names for IK. If None, uses defaults.
            gripper_joint_names: List of gripper joint names. If None, uses defaults.
            ik_scale: Scaling factor for IK joint velocity (0.0-1.0)
            ik_damping: Damping for singularity robustness (0.0-0.1)
        """
        self.model = model
        self.data = data

        # Convert string to RobotType enum if needed
        if isinstance(robot_type, str):
            robot_type = RobotType(robot_type)
        self.robot_type = robot_type

        self.ik_scale = ik_scale
        self.ik_damping = ik_damping

        # Get end effector site ID
        try:
            self.ee_site_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name
            )
        except Exception as e:
            raise ValueError(f"End effector site '{ee_site_name}' not found: {e}")

        # Set default joint names based on robot type
        if arm_joint_names is None:
            if robot_type == RobotType.WXAI:
                arm_joint_names = [f"joint_{i}" for i in range(6)]
            elif robot_type == RobotType.STATIONARY_AI:
                # For dual arm, this will be set per arm
                arm_joint_names = []
            elif robot_type == RobotType.MOBILE_AI:
                # For mobile base with dual arms
                arm_joint_names = []

        if gripper_joint_names is None:
            if robot_type == RobotType.WXAI:
                gripper_joint_names = ["left_carriage_joint"]
            elif robot_type == RobotType.STATIONARY_AI:
                gripper_joint_names = []
            elif robot_type == RobotType.MOBILE_AI:
                gripper_joint_names = []

        # Get joint IDs and indices for arm joints
        self.arm_joint_ids = []
        self.arm_qpos_indices = []
        self.arm_dof_indices = []
        self.arm_actuator_ids = []

        for joint_name in arm_joint_names if arm_joint_names is not None else []:
            try:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self.arm_joint_ids.append(joint_id)

                # Get qpos address for this joint
                qpos_adr = model.jnt_qposadr[joint_id]
                self.arm_qpos_indices.append(qpos_adr)

                # Get degree_of_freedom address for this joint
                dof_addr = model.jnt_dofadr[joint_id]
                self.arm_dof_indices.append(dof_addr)

                # Get actuator ID for this joint (assumes actuator name matches joint name)
                try:
                    actuator_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name
                    )
                    self.arm_actuator_ids.append(actuator_id)
                except Exception:
                    print(f"Warning: Actuator for joint '{joint_name}' not found")
                    self.arm_actuator_ids.append(-1)

            except Exception as e:
                print(f"Warning: Joint '{joint_name}' not found: {e}")

        # Get gripper joint IDs and indices
        self.gripper_joint_ids = []
        self.gripper_qpos_indices = []
        self.gripper_actuator_ids = []

        for joint_name in (
            gripper_joint_names if gripper_joint_names is not None else []
        ):
            try:
                joint_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
                )
                self.gripper_joint_ids.append(joint_id)

                qpos_adr = model.jnt_qposadr[joint_id]
                self.gripper_qpos_indices.append(qpos_adr)

                # Get actuator ID - gripper actuator name varies by robot type
                # WXAI: "left_gripper"
                # Stationary AI: "follower_left_gripper" or "follower_right_gripper"
                # Mobile AI: "follower_left/gripper" or "follower_right/gripper"
                if robot_type == RobotType.WXAI:
                    gripper_actuator_name = "left_gripper"
                elif robot_type == RobotType.STATIONARY_AI:
                    # Pattern: follower_left_left_carriage_joint -> follower_left_gripper
                    if "follower_left" in joint_name:
                        gripper_actuator_name = "follower_left_gripper"
                    else:
                        gripper_actuator_name = "follower_right_gripper"
                elif robot_type == RobotType.MOBILE_AI:
                    # Pattern: follower_left/left_carriage_joint -> follower_left/gripper
                    if "follower_left" in joint_name:
                        gripper_actuator_name = "follower_left/gripper"
                    else:
                        gripper_actuator_name = "follower_right/gripper"
                else:
                    # Fallback: try to infer from joint name
                    gripper_actuator_name = joint_name.replace(
                        "left_carriage_joint", "gripper"
                    )
                    gripper_actuator_name = gripper_actuator_name.replace(
                        "right_carriage_joint", "gripper"
                    )

                try:
                    actuator_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_ACTUATOR, gripper_actuator_name
                    )
                    self.gripper_actuator_ids.append(actuator_id)
                except Exception:
                    print(
                        f"Warning: Gripper actuator '{gripper_actuator_name}' not found"
                    )
                    self.gripper_actuator_ids.append(-1)

            except Exception as e:
                print(f"Warning: Gripper joint '{joint_name}' not found: {e}")

        self.num_arm_joints = len(self.arm_joint_ids)

        # Allocate Jacobian matrices
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))

    def get_ee_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current end effector pose.

        Returns:
            position: End effector position [x, y, z]
            orientation: End effector orientation as quaternion [w, x, y, z]
        """
        # Get site position
        position = self.data.site(self.ee_site_id).xpos.copy()

        # Get site orientation (rotation matrix)
        xmat = self.data.site(self.ee_site_id).xmat.reshape(3, 3)

        # Convert rotation matrix to quaternion
        rotation = Rotation.from_matrix(xmat)
        quat_xyzw = rotation.as_quat()  # scipy format: [x, y, z, w]
        orientation = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )  # [w, x, y, z]

        return position, orientation

    def get_arm_joint_positions(self) -> np.ndarray:
        """Get current arm joint positions.

        Returns:
            Joint positions array
        """
        return np.array([self.data.qpos[idx] for idx in self.arm_qpos_indices])

    def set_arm_joint_positions(self, joint_positions: np.ndarray) -> None:
        """Set arm joint target positions via actuators.

        Args:
            joint_positions: Target joint positions
        """
        for i, actuator_id in enumerate(self.arm_actuator_ids):
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = joint_positions[i]

    def get_gripper_position(self) -> float:
        """Get current gripper position.

        Returns:
            Gripper position (opening width in meters)
        """
        if len(self.gripper_qpos_indices) == 0:
            return 0.0
        return self.data.qpos[self.gripper_qpos_indices[0]]

    def set_gripper_position(self, position: float) -> None:
        """Set gripper target position via actuator.

        Args:
            position: Gripper opening width (0.022 = closed, 0.044 = open)
        """
        for actuator_id in self.gripper_actuator_ids:
            if actuator_id >= 0:
                self.data.ctrl[actuator_id] = position

    def open_gripper(self) -> None:
        """Open gripper to maximum width."""
        self.set_gripper_position(GRIPPER_OPEN_POSITION)

    def close_gripper(self) -> None:
        """Close gripper around object."""
        self.set_gripper_position(GRIPPER_CLOSED_POSITION)

    def compute_ik_step(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = None,
        position_only: bool = False,
    ) -> np.ndarray:
        """Compute one step of differential IK using damped least squares.

        Args:
            target_position: Target end effector position [x, y, z]
            target_orientation: Target orientation as quaternion [w, x, y, z].
                              If None, only position control is used.
            position_only: If True, ignore orientation even if provided

        Returns:
            Joint position delta to apply
        """
        # Get current end effector pose
        current_position, current_orientation = self.get_ee_pose()

        # Compute position error
        position_error = target_position - current_position

        # Compute orientation error if needed
        if target_orientation is not None and not position_only:
            # Orientation error using quaternion difference
            # e = 2 * (q_target * q_current^-1).xyz * sign(q_diff.w)
            q_target = target_orientation  # [w, x, y, z]
            q_current = current_orientation

            # Quaternion conjugate (inverse for unit quaternions)
            q_current_conj = np.array(
                [q_current[0], -q_current[1], -q_current[2], -q_current[3]]
            )

            # Quaternion multiplication: q_target * q_current_conj
            q_diff = self._quat_multiply(q_target, q_current_conj)

            # Orientation error
            orientation_error = 2 * q_diff[1:4] * np.sign(q_diff[0])

            # Combine position and orientation error
            error = np.concatenate([position_error, orientation_error])
        else:
            # Position-only control
            error = position_error

        # Get Jacobian for end effector site
        mujoco.mj_jacSite(
            self.model, self.data, self.jac_pos, self.jac_rot, self.ee_site_id
        )

        # Extract Jacobian for arm joints only
        if target_orientation is not None and not position_only:
            # Use both position and orientation Jacobian (6 x n_joints)
            J = np.vstack(
                [
                    self.jac_pos[:, self.arm_dof_indices],
                    self.jac_rot[:, self.arm_dof_indices],
                ]
            )
        else:
            # Use only position Jacobian (3 x n_joints)
            J = self.jac_pos[:, self.arm_dof_indices]

        # Damped least squares: J^T (J J^T + λ²I)^-1
        JJt = J @ J.T
        damping_matrix = self.ik_damping**2 * np.eye(JJt.shape[0])

        try:
            JJt_damped_inv = np.linalg.inv(JJt + damping_matrix)
            J_pinv = J.T @ JJt_damped_inv
        except np.linalg.LinAlgError:
            print("Warning: Singular matrix in IK computation, using larger damping")
            damping_matrix = (self.ik_damping * 10) ** 2 * np.eye(JJt.shape[0])
            JJt_damped_inv = np.linalg.inv(JJt + damping_matrix)
            J_pinv = J.T @ JJt_damped_inv

        # Compute joint velocity
        dq = self.ik_scale * J_pinv @ error

        return dq

    def set_ee_pose(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray = None,
        position_only: bool = False,
    ) -> float:
        """Move end effector toward target pose using IK.

        Computes and applies one IK step. Should be called repeatedly
        until target is reached.

        Args:
            target_position: Target position [x, y, z]
            target_orientation: Target orientation quaternion [w, x, y, z]
            position_only: If True, only control position

        Returns:
            Position error magnitude (meters)
        """
        # Compute IK step
        dq = self.compute_ik_step(target_position, target_orientation, position_only)

        # Get current joint positions
        current_q = self.get_arm_joint_positions()

        # Apply delta
        new_q = current_q + dq

        # Set new joint positions
        self.set_arm_joint_positions(new_q)

        # Return error for monitoring
        current_position, _ = self.get_ee_pose()
        error = np.linalg.norm(target_position - current_position)

        return error

    def _quat_multiply(self, q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        """Multiply two quaternions [w, x, y, z].

        Args:
            q1: First quaternion [w, x, y, z]
            q2: Second quaternion [w, x, y, z]

        Returns:
            Product quaternion [w, x, y, z]
        """
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return np.array([w, x, y, z])


class DualArmIKController:
    """IK controller for dual-arm robots (Stationary AI, Mobile AI).

    Manages two independent IK controllers, one for each arm.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_type: RobotType,
        left_ee_site_name: str = "left_ee_site",
        right_ee_site_name: str = "right_ee_site",
        left_arm_joint_names: list = None,
        right_arm_joint_names: list = None,
        left_gripper_joint_names: list = None,
        right_gripper_joint_names: list = None,
        ik_scale: float = DEFAULT_IK_SCALE,
        ik_damping: float = DEFAULT_IK_DAMPING,
    ):
        """Initialize dual-arm IK controller.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            robot_type: STATIONARY_AI or MOBILE_AI
            left_ee_site_name: Name of left arm end effector site
            right_ee_site_name: Name of right arm end effector site
            left_arm_joint_names: Left arm joint names
            right_arm_joint_names: Right arm joint names
            left_gripper_joint_names: Left gripper joint names
            right_gripper_joint_names: Right gripper joint names
            ik_scale: IK scaling factor
            ik_damping: IK damping factor
        """
        self.model = model
        self.data = data
        self.robot_type = robot_type

        # Create left arm controller
        self.left_controller = IKController(
            model=model,
            data=data,
            robot_type=robot_type,
            ee_site_name=left_ee_site_name,
            arm_joint_names=left_arm_joint_names,
            gripper_joint_names=left_gripper_joint_names,
            ik_scale=ik_scale,
            ik_damping=ik_damping,
        )

        # Create right arm controller
        self.right_controller = IKController(
            model=model,
            data=data,
            robot_type=robot_type,
            ee_site_name=right_ee_site_name,
            arm_joint_names=right_arm_joint_names,
            gripper_joint_names=right_gripper_joint_names,
            ik_scale=ik_scale,
            ik_damping=ik_damping,
        )
