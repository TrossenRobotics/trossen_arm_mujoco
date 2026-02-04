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
"""Mobile AI Sequential Pick-and-Place Demonstration.

Demonstrates mobile base motion and dual-arm pick-and-place using the Mobile AI robot.

Usage:
    python trossen_arm_mujoco/scripts/mobile_ai_pick_place.py
"""

from __future__ import annotations

import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.ik_controller import IKController, RobotType

# Default configuration constants
DEFAULT_CUBE_POSITION = np.array([1.0, 0.3, 0.825])
DEFAULT_CUBE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_INTERMEDIATE_POSITION = np.array([1.0, 0.0, 0.825])
DEFAULT_TARGET_POSITION = np.array([1.0, -0.3, 0.825])

# Home positions for arms
LEFT_ARM_HOME_POSITION = np.array([1.0, 0.3, 1.1])
RIGHT_ARM_HOME_POSITION = np.array([1.0, -0.3, 1.1])

# Phase timing (steps)
LEFT_ARM_EVENTS_DT = [400, 300, 100, 300, 400, 300, 100, 300, 400]
RIGHT_ARM_EVENTS_DT = [400, 300, 100, 300, 400, 300, 100, 300, 400]

# Mobile base movement
BASE_INITIAL_POSITION = np.array([0.0, 0.0, 0.0])
BASE_TARGET_POSITION = np.array([0.3, 0.0, 0.0])
BASE_MOVEMENT_STEPS = 600

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15
APPROACH_OFFSET = np.array([0.0, 0.0, 0.0])
DOWNWARD_ORIENTATION = np.array([0.70710678, 0.0, 0.70710678, 0.0])

# Scene configuration
SCENE_XML_PATH = "trossen_arm_mujoco/assets/mobile_ai/scene_mobile_ai_pick_place.xml"

# Robot controller configuration
LEFT_ARM_JOINT_NAMES = [f"follower_left/joint_{i}" for i in range(6)]
LEFT_GRIPPER_JOINT_NAMES = ["follower_left/left_carriage_joint"]
RIGHT_ARM_JOINT_NAMES = [f"follower_right/joint_{i}" for i in range(6)]
RIGHT_GRIPPER_JOINT_NAMES = ["follower_right/left_carriage_joint"]

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03
POSITION_THRESHOLD = 0.04
MAX_STEPS_PER_WAYPOINT = 100


class MobileAIPickPlace:
    """Sequential dual-arm pick-and-place for Mobile AI robot."""

    def __init__(
        self,
        left_events_dt: list[int] | None = None,
        right_events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        intermediate_position: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
    ):
        """Initialize sequential dual-arm pick-and-place task.

        Args:
            left_events_dt: Time deltas for left arm phases.
            right_events_dt: Time deltas for right arm phases.
            cube_initial_position: Initial cube position.
            cube_initial_orientation: Initial cube orientation.
            intermediate_position: Middle position where left places/right picks.
            target_position: Final target position.
        """
        self.cube_initial_position = (
            cube_initial_position
            if cube_initial_position is not None
            else DEFAULT_CUBE_POSITION.copy()
        )
        self.cube_initial_orientation = (
            cube_initial_orientation
            if cube_initial_orientation is not None
            else DEFAULT_CUBE_ORIENTATION.copy()
        )
        self.intermediate_position = (
            intermediate_position
            if intermediate_position is not None
            else DEFAULT_INTERMEDIATE_POSITION.copy()
        )
        self.target_position = (
            target_position
            if target_position is not None
            else DEFAULT_TARGET_POSITION.copy()
        )

        self.left_events_dt = (
            left_events_dt if left_events_dt is not None else LEFT_ARM_EVENTS_DT.copy()
        )
        self.right_events_dt = (
            right_events_dt
            if right_events_dt is not None
            else RIGHT_ARM_EVENTS_DT.copy()
        )

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET.copy()
        self.left_home = LEFT_ARM_HOME_POSITION.copy()
        self.right_home = RIGHT_ARM_HOME_POSITION.copy()
        self.downward_orientation = DOWNWARD_ORIENTATION.copy()

        # MuJoCo components
        self.model = None
        self.data = None
        self.left_robot = None
        self.right_robot = None
        self.cube_body_id = None

        # Trajectory state
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0
        self.current_phase = (
            "driving"  # "driving", "left_arm_pickup", or "right_arm_pickup"
        )
        self.drive_step_counter = 0
        self.mobile_base_joint_id = None

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Initialize left arm controller
        self.left_robot = IKController(
            model=self.model,
            data=self.data,
            robot_type=RobotType.MOBILE_AI,
            ee_site_name="follower_left/ee_site",
            arm_joint_names=LEFT_ARM_JOINT_NAMES,
            gripper_joint_names=LEFT_GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Initialize right arm controller
        self.right_robot = IKController(
            model=self.model,
            data=self.data,
            robot_type=RobotType.MOBILE_AI,
            ee_site_name="follower_right/ee_site",
            arm_joint_names=RIGHT_ARM_JOINT_NAMES,
            gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

        # Get mobile base joint ID for base movement (freejoint at qpos_addr 0)
        self.mobile_base_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "mobile_base_freejoint"
        )
        assert self.model is not None
        if self.mobile_base_joint_id >= 0:
            self.mobile_base_joint_id = self.model.jnt_qposadr[
                self.mobile_base_joint_id
            ]
        else:
            print("Warning: Could not find mobile_base_freejoint")

    def forward(self) -> bool:
        """Execute one simulation step of the sequential pick-and-place.

        Returns:
            bool: True if task is in progress, False if complete.
        """
        assert self.data is not None
        if self.is_done():
            return False

        # Phase 1: Driving
        if self.current_phase == "driving":
            if self.drive_step_counter < BASE_MOVEMENT_STEPS:
                # Smooth interpolation
                alpha = self.drive_step_counter / BASE_MOVEMENT_STEPS
                current_pos = BASE_INITIAL_POSITION + alpha * (
                    BASE_TARGET_POSITION - BASE_INITIAL_POSITION
                )

                # Set mobile base position
                self.data.qpos[
                    self.mobile_base_joint_id : self.mobile_base_joint_id + 3
                ] = current_pos

                self.drive_step_counter += 1
            else:
                print("Driving complete!")
                self.current_phase = "left_arm_pickup"
                self.trajectory_index = 0
                self.waypoint_step_count = 0

            return True

        # Generate trajectories on first call
        if self.left_trajectory is None:
            self.generate_left_arm_trajectory()
        if self.right_trajectory is None:
            self.generate_right_arm_trajectory()

        assert self.left_robot is not None
        assert self.right_robot is not None
        assert self.left_trajectory is not None
        assert self.right_trajectory is not None

        # Phase 2: Left arm pickup
        if self.current_phase == "left_arm_pickup":
            if self.trajectory_index < len(self.left_trajectory):
                goal_pos, goal_ori, _ = self.left_trajectory[self.trajectory_index]
                error = self.left_robot.set_ee_pose(
                    target_position=goal_pos,
                    target_orientation=goal_ori,
                    position_only=False,
                )

                self.waypoint_step_count += 1

                if (
                    error < POSITION_THRESHOLD
                    or self.waypoint_step_count >= MAX_STEPS_PER_WAYPOINT
                ):
                    self.trajectory_index += 1
                    self.waypoint_step_count = 0

                    # Calculate phase boundaries
                    phase_boundaries = [0]
                    cumulative = 0
                    for duration in self.left_events_dt:
                        cumulative += duration
                        phase_boundaries.append(cumulative)

                    # Gripper control
                    if (
                        phase_boundaries[2]
                        <= self.trajectory_index
                        < phase_boundaries[3]
                    ):
                        self.left_robot.close_gripper()
                    elif (
                        phase_boundaries[6]
                        <= self.trajectory_index
                        < phase_boundaries[7]
                    ):
                        self.left_robot.open_gripper()

                # Check if left arm trajectory is complete
                if self.trajectory_index >= len(self.left_trajectory):
                    print("Left arm complete!")
                    self.current_phase = "right_arm_pickup"
                    self.trajectory_index = 0
                    self.waypoint_step_count = 0

        # Phase 3: Right arm pickup
        elif self.current_phase == "right_arm_pickup":
            if self.trajectory_index < len(self.right_trajectory):
                goal_pos, goal_ori, _ = self.right_trajectory[self.trajectory_index]
                error = self.right_robot.set_ee_pose(
                    target_position=goal_pos,
                    target_orientation=goal_ori,
                    position_only=False,
                )

                self.waypoint_step_count += 1

                if (
                    error < POSITION_THRESHOLD
                    or self.waypoint_step_count >= MAX_STEPS_PER_WAYPOINT
                ):
                    self.trajectory_index += 1
                    self.waypoint_step_count = 0

                    # Calculate phase boundaries
                    phase_boundaries = [0]
                    cumulative = 0
                    for duration in self.right_events_dt:
                        cumulative += duration
                        phase_boundaries.append(cumulative)

                    # Gripper control
                    if (
                        phase_boundaries[2]
                        <= self.trajectory_index
                        < phase_boundaries[3]
                    ):
                        self.right_robot.close_gripper()
                    elif (
                        phase_boundaries[6]
                        <= self.trajectory_index
                        < phase_boundaries[7]
                    ):
                        self.right_robot.open_gripper()

        return True

    def is_done(self) -> bool:
        """Check if sequential task is complete."""
        return (
            self.current_phase == "right_arm_pickup"
            and self.right_trajectory is not None
            and self.trajectory_index >= len(self.right_trajectory)
        )

    def reset(
        self,
        cube_position: np.ndarray | None = None,
        cube_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset task to initial state."""
        self.reset_robot()
        self.reset_cube(position=cube_position, orientation=cube_orientation)

    def reset_robot(self) -> None:
        """Reset both arms and clear trajectories."""
        if self.left_robot is None or self.right_robot is None:
            raise RuntimeError("Cannot reset robot: controllers not initialized.")

        # Reset arm joints
        for joint_name in LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = 0.0

        # Open grippers
        self.left_robot.open_gripper()
        self.right_robot.open_gripper()

        # Clear trajectories
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0
        self.current_phase = "driving"
        self.drive_step_counter = 0

        mujoco.mj_forward(self.model, self.data)

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ) -> None:
        """Reset cube to specified or initial pose."""
        if self.cube_body_id is None:
            raise RuntimeError("Cannot reset cube: cube not initialized.")

        reset_position = (
            position if position is not None else self.cube_initial_position
        )
        reset_orientation = (
            orientation if orientation is not None else self.cube_initial_orientation
        )

        cube_joint_id = None
        for i in range(self.model.njnt):
            if self.model.jnt_bodyid[i] == self.cube_body_id:
                cube_joint_id = i
                break

        if cube_joint_id is None:
            raise RuntimeError("Could not find joint for cube body")

        qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.data.qpos[qpos_addr : qpos_addr + 3] = reset_position
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = reset_orientation

        qvel_addr = self.model.jnt_dofadr[cube_joint_id]
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def get_cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current cube position and orientation."""
        assert self.data is not None
        assert self.cube_body_id is not None
        cube_pos = self.data.xpos[self.cube_body_id].copy()
        cube_xmat = self.data.xmat[self.cube_body_id].reshape(3, 3)
        rotation = Rotation.from_matrix(cube_xmat)
        quat_xyzw = rotation.as_quat()
        cube_quat = np.array([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]])
        return cube_pos, cube_quat

    def make_trajectory(
        self,
        key_frames: list[np.ndarray],
        orientations: list[np.ndarray],
        dt: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Generate smooth trajectory via linear interpolation between keyframes."""
        if len(key_frames) != len(dt) + 1:
            raise ValueError(f"Expected {len(dt) + 1} keyframes for {len(dt)} segments")
        if len(orientations) != len(key_frames):
            raise ValueError("Orientations must match keyframe count")

        trajectory = []
        cumulative_step = 0

        for i in range(len(dt)):
            start_pos = np.array(key_frames[i], dtype=np.float64)
            end_pos = np.array(key_frames[i + 1], dtype=np.float64)
            start_ori = np.array(orientations[i], dtype=np.float64)
            end_ori = np.array(orientations[i + 1], dtype=np.float64)
            n_steps = dt[i]

            rot_start = Rotation.from_quat(
                [start_ori[1], start_ori[2], start_ori[3], start_ori[0]]
            )
            rot_end = Rotation.from_quat(
                [end_ori[1], end_ori[2], end_ori[3], end_ori[0]]
            )

            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)

                if alpha == 0.0:
                    interpolated_ori = start_ori
                else:
                    # Normalized linear interpolation for quaternions
                    q = rot_start.as_quat() * (1 - alpha) + rot_end.as_quat() * alpha
                    norm = np.linalg.norm(q)
                    if norm > 0.0:
                        q = q / norm
                    rot_interp = Rotation.from_quat(q)
                    quat_xyzw = rot_interp.as_quat()
                    interpolated_ori = np.array(
                        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                    )

                trajectory.append(
                    (interpolated_pos, interpolated_ori, cumulative_step + step)
                )

            cumulative_step += n_steps

        trajectory.append(
            (
                np.array(key_frames[-1], dtype=np.float64),
                np.array(orientations[-1], dtype=np.float64),
                cumulative_step,
            )
        )

        return trajectory

    def generate_left_arm_trajectory(self) -> None:
        """Generate left arm pick-place trajectory.

        Left arm picks from initial position and places at intermediate position.
        9-phase trajectory:
        0. Move to pre-pick position
        1. Descend to approach
        2. Close gripper
        3. Lift with cube
        4. Move to pre-place
        5. Descend to place
        6. Open gripper
        7. Retreat
        8. Return home
        """
        cube_pos, _ = self.get_cube_pose()
        place_pos = self.intermediate_position
        assert self.left_robot is not None
        left_ee_pos, _ = self.left_robot.get_ee_pose()

        key_frames = [
            left_ee_pos,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            cube_pos + self.approach_offset,
            cube_pos + self.approach_offset,  # Grasp
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + self.approach_offset,
            place_pos + self.approach_offset,  # Release
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.left_home,
        ]

        orientations = [self.downward_orientation for _ in key_frames]

        self.left_trajectory = self.make_trajectory(
            key_frames, orientations, self.left_events_dt
        )

    def generate_right_arm_trajectory(self) -> None:
        """Generate right arm pick-place trajectory.

        Right arm picks from intermediate position and places at target.
        """
        pick_pos = self.intermediate_position
        place_pos = self.target_position
        assert self.right_robot is not None
        right_ee_pos, _ = self.right_robot.get_ee_pose()

        key_frames = [
            right_ee_pos,
            pick_pos + np.array([0.0, 0.0, self.clearance_height]),
            pick_pos + self.approach_offset,
            pick_pos + self.approach_offset,  # Grasp
            pick_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            place_pos + self.approach_offset,
            place_pos + self.approach_offset,  # Release
            place_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.right_home,
        ]

        orientations = [self.downward_orientation for _ in key_frames]

        self.right_trajectory = self.make_trajectory(
            key_frames, orientations, self.right_events_dt
        )


def main():
    """Main execution function."""
    print("=" * 70)
    print("Mobile AI Sequential Dual-Arm Pick-and-Place Demo (MuJoCo)")
    print("=" * 70)
    print()
    print("The left arm picks up the cube and places it at intermediate position,")
    print("then the right arm picks it up and places it at the target.")
    print()
    print("Task sequence:")
    print("  1. Left arm: Pick cube from left side")
    print("  2. Left arm: Place cube at center")
    print("  3. Right arm: Pick cube from center")
    print("  4. Right arm: Place cube at right side")
    print()
    print("=" * 70)
    print()

    pick_place = MobileAIPickPlace()
    pick_place.setup_scene()
    pick_place.reset()

    task_completed = False

    with mujoco.viewer.launch_passive(pick_place.model, pick_place.data) as viewer:
        # Set viewer camera for better initial view
        viewer.cam.lookat[:] = [1.0, 0.0, 0.9]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        dt = pick_place.model.opt.timestep

        while viewer.is_running():
            step_start = time.time()

            if not task_completed:
                pick_place.forward()

            if pick_place.is_done() and not task_completed:
                print("\n✓ Task complete!")
                print("\nSequential dual-arm pick-and-place finished successfully.")
                task_completed = True

            mujoco.mj_step(pick_place.model, pick_place.data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\n✓ Mobile AI pick-and-place demo completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping dual-arm pick-and-place demo...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
