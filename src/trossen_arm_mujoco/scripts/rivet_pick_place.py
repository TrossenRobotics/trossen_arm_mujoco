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
"""Rivet Sequential Pick-and-Place Demonstration.

Demonstrates lift motion, mobile base motion, and dual-arm pick-and-place using the Rivet robot.

Usage:
    uv run rivet_pick_place
"""

from __future__ import annotations

import os
import sys
import time

import mujoco
import mujoco.viewer
import numpy as np

from trossen_arm_mujoco import ASSETS_DIR
from trossen_arm_mujoco.base_controller import BaseController
from trossen_arm_mujoco.controller import Controller, RobotType

# Default configuration constants
DEFAULT_CUBE_POSITION = np.array([1.0, 0.3, 0.845])
DEFAULT_CUBE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_INTERMEDIATE_POSITION = np.array([1.0, 0.0, 0.845])
DEFAULT_TARGET_POSITION = np.array([1.0, -0.3, 0.845])

# Home positions for arms
LEFT_ARM_HOME_POSITION = np.array([1.0, 0.3, 1.1])
RIGHT_ARM_HOME_POSITION = np.array([1.0, -0.3, 1.1])

# Phase timing (steps)
LEFT_ARM_EVENTS_DT = [1200, 300, 100, 300, 400, 300, 100, 300, 400]
RIGHT_ARM_EVENTS_DT = [1200, 300, 100, 300, 400, 300, 100, 300, 400]

# Mobile base movement
BASE_TARGET_POSITION = np.array([0.5, 0.0])
BASE_SPEED = 0.3  # m/s
BASE_DECELERATION = 1.0  # m/s^2
BASE_POSITION_THRESHOLD = 0.005  # m
BASE_VELOCITY_THRESHOLD = 0.002  # m/s

# Lift movement
LIFT_HEIGHT = 0.5795  # m
LIFT_SPEED = 0.2  # m/s
LIFT_POSITION_THRESHOLD = 0.002  # m

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15
APPROACH_OFFSET = np.array([0.0, 0.0, 0.0])
DOWNWARD_ORIENTATION = np.array([0.70710678, 0.0, 0.70710678, 0.0])

# Scene configuration
SCENE_XML_PATH = os.path.join(ASSETS_DIR, "rivet", "scene_rivet_pick_place.xml")

# Robot controller configuration
LEFT_ARM_JOINT_NAMES = [f"follower_left_joint_{i}" for i in range(6)]
LEFT_GRIPPER_JOINT_NAMES = ["follower_left_left_carriage_joint"]
RIGHT_ARM_JOINT_NAMES = [f"follower_right_joint_{i}" for i in range(6)]
RIGHT_GRIPPER_JOINT_NAMES = ["follower_right_left_carriage_joint"]
FINGER_JOINT_NAMES = [
    f"follower_{side}_{finger}_carriage_joint"
    for side in ("left", "right")
    for finger in ("left", "right")
]
GRIPPER_OPEN_POSITION = 0.044

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03
POSITION_THRESHOLD = 0.04
MAX_STEPS_PER_WAYPOINT = 100


class RivetPickPlace:
    """Sequential dual-arm pick-and-place for Rivet robot."""

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

        :param left_events_dt: Time deltas for left arm phases.
        :param right_events_dt: Time deltas for right arm phases.
        :param cube_initial_position: Initial cube position.
        :param cube_initial_orientation: Initial cube orientation.
        :param intermediate_position: Middle position where left places/right picks.
        :param target_position: Final target position.
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
            target_position if target_position is not None else DEFAULT_TARGET_POSITION.copy()
        )

        self.left_events_dt = (
            left_events_dt if left_events_dt is not None else LEFT_ARM_EVENTS_DT.copy()
        )
        self.right_events_dt = (
            right_events_dt if right_events_dt is not None else RIGHT_ARM_EVENTS_DT.copy()
        )

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET.copy()
        self.left_home = LEFT_ARM_HOME_POSITION.copy()
        self.right_home = RIGHT_ARM_HOME_POSITION.copy()
        self.downward_orientation = DOWNWARD_ORIENTATION.copy()

        # MuJoCo components
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.base: BaseController | None = None
        self.left_robot: Controller | None = None
        self.right_robot: Controller | None = None
        self.cube_body_id: int | None = None

        # Trajectory state
        self.left_trajectory: list[tuple[np.ndarray, np.ndarray, int]] | None = None
        self.right_trajectory: list[tuple[np.ndarray, np.ndarray, int]] | None = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0
        # "lifting", "driving", "left_arm_pickup", or "right_arm_pickup"
        self.current_phase = "lifting"
        self.base_body_id: int | None = None

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Initialize base controller
        self.base = BaseController(model=self.model, data=self.data)

        # Initialize left arm controller
        self.left_robot = Controller(
            model=self.model,
            data=self.data,
            robot_type=RobotType.RIVET,
            ee_site_name="follower_left_ee_site",
            arm_joint_names=LEFT_ARM_JOINT_NAMES,
            gripper_joint_names=LEFT_GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Initialize right arm controller
        self.right_robot = Controller(
            model=self.model,
            data=self.data,
            robot_type=RobotType.RIVET,
            ee_site_name="follower_right_ee_site",
            arm_joint_names=RIGHT_ARM_JOINT_NAMES,
            gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

        # Get base body ID
        self.base_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    def forward(self) -> bool:
        """Execute one simulation step of the sequential pick-and-place.

        :return: True if task is in progress, False if complete.
        """
        assert self.data is not None
        if self.is_done():
            return False

        assert self.base is not None

        # Run the base every step so it ramps and then holds still while the arms work
        self.base.update()

        # Phase 1: Lifting
        if self.current_phase == "lifting":
            self.base.set_lift_position(LIFT_HEIGHT, LIFT_SPEED)
            if abs(self.base.get_lift_position() - LIFT_HEIGHT) < LIFT_POSITION_THRESHOLD:
                print("Lifting complete!")
                self.current_phase = "driving"

            return True

        # Phase 2: Driving
        if self.current_phase == "driving":
            if self.drive_to(BASE_TARGET_POSITION):
                print("Driving complete!")
                self.current_phase = "left_arm_pickup"
                self.trajectory_index = 0
                self.waypoint_step_count = 0

            return True

        # Generate each arm's trajectory when its phase starts, from where the arm is then
        if self.left_trajectory is None:
            self.generate_left_arm_trajectory()
        if self.current_phase == "right_arm_pickup" and self.right_trajectory is None:
            self.generate_right_arm_trajectory()

        assert self.left_robot is not None
        assert self.right_robot is not None
        assert self.left_trajectory is not None

        # Phase 3: Left arm pickup
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
                    if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                        self.left_robot.close_gripper()
                    elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                        self.left_robot.open_gripper()

                # Check if left arm trajectory is complete
                if self.trajectory_index >= len(self.left_trajectory):
                    print("Left arm complete!")
                    self.current_phase = "right_arm_pickup"
                    self.trajectory_index = 0
                    self.waypoint_step_count = 0

        # Phase 4: Right arm pickup
        elif self.current_phase == "right_arm_pickup":
            assert self.right_trajectory is not None
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
                    if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                        self.right_robot.close_gripper()
                    elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                        self.right_robot.open_gripper()

        return True

    def drive_to(self, target_position: np.ndarray) -> bool:
        """Drive the base toward a position on the floor, slowing down near it.

        :param target_position: Target base position [x, y] in the world frame.
        :return: True once the base has reached the position.
        """
        assert self.data is not None
        assert self.base is not None

        error = target_position - self.data.xpos[self.base_body_id][:2]
        distance = np.linalg.norm(error)
        if distance < BASE_POSITION_THRESHOLD:
            # Done once the base has come to a stop
            self.base.stop()
            return bool(np.linalg.norm(self.base.get_velocity()) < BASE_VELOCITY_THRESHOLD)

        # Slow down so the base can stop in the remaining distance
        speed = min(BASE_SPEED, np.sqrt(2 * BASE_DECELERATION * distance))

        # Rotate the world frame direction into the base frame
        rotation = self.data.xmat[self.base_body_id].reshape(3, 3)[:2, :2]
        velocity = rotation.T @ (error / distance * speed)
        self.base.set_velocity(velocity[0], velocity[1], 0.0)
        return False

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
        """Reset base, lift, and both arms and clear trajectories."""
        if self.left_robot is None or self.right_robot is None or self.base is None:
            raise RuntimeError("Cannot reset robot: controllers not initialized.")
        assert self.model is not None
        assert self.data is not None

        # Reset arm joints
        for joint_name in LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = 0.0

        # Lower the lift, then stop the base and hold the lift there
        self.data.qpos[self.base.lift_qpos_index] = 0.0
        self.base.reset()

        # Reset grippers to open
        for joint_name in FINGER_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            self.data.qpos[self.model.jnt_qposadr[joint_id]] = GRIPPER_OPEN_POSITION
        self.left_robot.open_gripper()
        self.right_robot.open_gripper()

        # Clear trajectories
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0
        self.current_phase = "lifting"

        mujoco.mj_forward(self.model, self.data)

    def reset_cube(
        self, position: np.ndarray | None = None, orientation: np.ndarray | None = None
    ) -> None:
        """Reset cube to specified or initial pose."""
        if self.cube_body_id is None:
            raise RuntimeError("Cannot reset cube: cube not initialized.")
        assert self.model is not None
        assert self.data is not None

        reset_position = position if position is not None else self.cube_initial_position
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
        cube_quat = self.data.xquat[self.cube_body_id].copy()
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

            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)

                if alpha == 0.0:
                    interpolated_ori = start_ori
                else:
                    # Normalized linear interpolation for quaternions
                    q = start_ori * (1 - alpha) + end_ori * alpha
                    norm = np.linalg.norm(q)
                    if norm > 0.0:
                        q = q / norm
                    interpolated_ori = q

                trajectory.append((interpolated_pos, interpolated_ori, cumulative_step + step))

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
        left_ee_pos, left_ee_ori = self.left_robot.get_ee_pose()

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

        # Start from the current orientation and turn downward during the first segment
        if np.dot(left_ee_ori, self.downward_orientation) < 0.0:
            left_ee_ori = -left_ee_ori
        orientations = [left_ee_ori] + [self.downward_orientation for _ in key_frames[1:]]

        self.left_trajectory = self.make_trajectory(key_frames, orientations, self.left_events_dt)

    def generate_right_arm_trajectory(self) -> None:
        """Generate right arm pick-place trajectory.

        Right arm picks from intermediate position and places at target.
        """
        pick_pos, _ = self.get_cube_pose()
        place_pos = self.target_position
        assert self.right_robot is not None
        right_ee_pos, right_ee_ori = self.right_robot.get_ee_pose()

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

        # Start from the current orientation and turn downward during the first segment
        if np.dot(right_ee_ori, self.downward_orientation) < 0.0:
            right_ee_ori = -right_ee_ori
        orientations = [right_ee_ori] + [self.downward_orientation for _ in key_frames[1:]]

        self.right_trajectory = self.make_trajectory(
            key_frames, orientations, self.right_events_dt
        )


def main():
    print("Running Rivet pick-and-place demo. Close the viewer to exit.")
    pick_place = RivetPickPlace()
    pick_place.setup_scene()
    pick_place.reset()

    task_completed = False

    with mujoco.viewer.launch_passive(pick_place.model, pick_place.data) as viewer:
        # Set viewer camera for better initial view
        viewer.cam.lookat[:] = [0.8, 0.0, 0.7]
        viewer.cam.distance = 2.8
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

        dt = pick_place.model.opt.timestep

        while viewer.is_running():
            step_start = time.time()

            if not task_completed:
                pick_place.forward()

            if pick_place.is_done() and not task_completed:
                print("\nTask complete!")
                print("\nSequential dual-arm pick-and-place finished successfully.")
                task_completed = True

            mujoco.mj_step(pick_place.model, pick_place.data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\nRivet pick-and-place demo completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping dual-arm pick-and-place demo...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
