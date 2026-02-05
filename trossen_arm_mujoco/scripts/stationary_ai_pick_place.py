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
"""Stationary AI Pick-and-Place Demonstration.

Demonstrates dual-arm pick-and-place with handoff using the Stationary AI robot.

Usage:
    python trossen_arm_mujoco/scripts/stationary_ai_pick_place.py
"""

from __future__ import annotations

import sys
import time

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

from trossen_arm_mujoco.src.controller import Controller, RobotType

# Default configuration constants
DEFAULT_CUBE_POSITION = np.array([0.0, 0.25, 0.045])
DEFAULT_CUBE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_TARGET_POSITION = np.array([0.0, -0.25, 0.025])
CENTER_HANDOFF_POSITION = np.array([0.0, 0.0, 0.25])
LEFT_ARM_HOME_POSITION = np.array([0.0, 0.3, 0.3])
RIGHT_ARM_HOME_POSITION = np.array([0.0, -0.3, 0.3])

# Phase timing (steps) - 15 phases for dual-arm handoff
DEFAULT_EVENTS_DT = [
    600,  # 0: Left to pre-pick
    300,  # 1: Left rotate at pre-pick
    200,  # 2: Left descend to pick
    200,  # 3: Left grasp (increased hold time for secure grasp)
    400,  # 4: Left lift
    600,  # 5: Both to handoff
    300,  # 6: Right approach
    200,  # 7: Right grasp (increased hold time for secure grasp)
    200,  # 8: Left release (increased hold time to ensure transfer)
    400,  # 9: Left retreat
    400,  # 10: Right lift
    600,  # 11: Right to pre-place
    400,  # 12: Right descend
    200,  # 13: Right release
    600,  # 14: Both return home
]

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15
APPROACH_OFFSET = np.array([0.0, 0.0, 0.0])
HANDOFF_OFFSET = 0.00

# Gripper orientations (quaternion [w, x, y, z])
LEFT_ARM_DOWNWARD_ORIENTATION = np.array([0.5, 0.5, 0.5, -0.5])
RIGHT_ARM_DOWNWARD_ORIENTATION = np.array([0.5, -0.5, 0.5, 0.5])
LEFT_ARM_HANDOFF_ORIENTATION = np.array([0.7071068, 0.0, 0.0, -0.7071068])
RIGHT_ARM_HANDOFF_ORIENTATION = np.array([0.7071068, 0.0, 0.0, 0.7071068])
RIGHT_ARM_RECEIVE_ORIENTATION = np.array([0.5, 0.5, 0.5, 0.5])

# Scene configuration
SCENE_XML_PATH = "trossen_arm_mujoco/assets/stationary_ai/scene_stationary_ai_pick_place.xml"

# Robot controller configuration
LEFT_ARM_JOINT_NAMES = [f"follower_left_joint_{i}" for i in range(6)]
LEFT_GRIPPER_JOINT_NAMES = ["follower_left_left_carriage_joint"]
RIGHT_ARM_JOINT_NAMES = [f"follower_right_joint_{i}" for i in range(6)]
RIGHT_GRIPPER_JOINT_NAMES = ["follower_right_left_carriage_joint"]

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03
POSITION_THRESHOLD = 0.12  # Tuned for IK controller with 2x orientation error
MAX_STEPS_PER_WAYPOINT = 250


class StationaryAIPickPlace:
    """Dual-arm pick-and-place with handoff for Stationary AI robot."""

    def __init__(
        self,
        events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
        handoff_position: np.ndarray | None = None,
    ):
        """Initialize dual-arm pick-and-place task.

        Args:
            events_dt: Time deltas for each phase of the task.
            cube_initial_position: Initial cube position [x, y, z].
            cube_initial_orientation: Initial cube orientation [w, x, y, z].
            target_position: Target place position [x, y, z].
            handoff_position: Center position for handoff [x, y, z].
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
        self.target_position = (
            target_position if target_position is not None else DEFAULT_TARGET_POSITION.copy()
        )
        self.handoff_position = (
            handoff_position if handoff_position is not None else CENTER_HANDOFF_POSITION.copy()
        )
        self.events_dt = events_dt if events_dt is not None else DEFAULT_EVENTS_DT.copy()

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET.copy()
        self.handoff_offset = HANDOFF_OFFSET
        self.left_home = LEFT_ARM_HOME_POSITION.copy()
        self.right_home = RIGHT_ARM_HOME_POSITION.copy()

        # MuJoCo components
        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.left_robot: Controller | None = None
        self.right_robot: Controller | None = None
        self.cube_body_id: int | None = None

        # Trajectory state
        self.left_trajectory: list[tuple[np.ndarray, np.ndarray, int]] | None = None
        self.right_trajectory: list[tuple[np.ndarray, np.ndarray, int]] | None = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Initialize left arm controller
        self.left_robot = Controller(
            model=self.model,
            data=self.data,
            robot_type=RobotType.STATIONARY_AI,
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
            robot_type=RobotType.STATIONARY_AI,
            ee_site_name="follower_right_ee_site",
            arm_joint_names=RIGHT_ARM_JOINT_NAMES,
            gripper_joint_names=RIGHT_GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")

    def forward(self) -> bool:
        """Execute one simulation step of the dual-arm handoff sequence.

        Returns:
            bool: True if sequence is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.left_trajectory is None or self.right_trajectory is None:
            self.generate_dual_arm_trajectory()

        assert self.left_trajectory is not None
        assert self.right_trajectory is not None
        assert self.left_robot is not None
        assert self.right_robot is not None

        if self.trajectory_index < len(self.left_trajectory):
            # Get target poses for both arms
            left_pos, left_ori, _ = self.left_trajectory[self.trajectory_index]
            right_pos, right_ori, _ = self.right_trajectory[self.trajectory_index]

            # Apply IK for both arms
            left_error = self.left_robot.set_ee_pose(
                target_position=left_pos,
                target_orientation=left_ori,
                position_only=False,
            )
            right_error = self.right_robot.set_ee_pose(
                target_position=right_pos,
                target_orientation=right_ori,
                position_only=False,
            )

            self.waypoint_step_count += 1

            # Calculate phase boundaries
            phase_boundaries = [0]
            cumulative = 0
            for duration in self.events_dt:
                cumulative += duration
                phase_boundaries.append(cumulative)

            # Track both arms during coordination, only active arm during solo phases
            if self.trajectory_index >= phase_boundaries[10]:
                relevant_error = right_error  # Right arm solo
            else:
                relevant_error = max(left_error, right_error)  # Both arms

            # Advance waypoint
            if (
                relevant_error < POSITION_THRESHOLD
                or self.waypoint_step_count >= MAX_STEPS_PER_WAYPOINT
            ):
                self.trajectory_index += 1
                self.waypoint_step_count = 0

                # Gripper control based on phase
                # Phase 3: Left grasps cube
                if phase_boundaries[3] <= self.trajectory_index < phase_boundaries[4]:
                    self.left_robot.close_gripper()
                # Phase 7: Right grasps cube
                elif phase_boundaries[7] <= self.trajectory_index < phase_boundaries[8]:
                    self.right_robot.close_gripper()
                # Phase 8: Left releases cube
                elif phase_boundaries[8] <= self.trajectory_index < phase_boundaries[9]:
                    self.left_robot.open_gripper()
                # Phase 13: Right releases cube
                elif phase_boundaries[13] <= self.trajectory_index < phase_boundaries[14]:
                    self.right_robot.open_gripper()

        return True

    def is_done(self) -> bool:
        """Check if dual-arm task is complete."""
        return self.left_trajectory is not None and self.trajectory_index >= len(
            self.left_trajectory
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
        """Reset both arms to home pose and clear trajectories."""
        if self.left_robot is None or self.right_robot is None:
            raise RuntimeError("Cannot reset robot: controllers not initialized.")
        assert self.model is not None
        assert self.data is not None

        # Reset all arm joints to zero
        for joint_name in LEFT_ARM_JOINT_NAMES + RIGHT_ARM_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = 0.0

        # Open both grippers
        self.left_robot.open_gripper()
        self.right_robot.open_gripper()

        # Clear trajectories
        self.left_trajectory = None
        self.right_trajectory = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0

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

        # Find cube freejoint
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
            rot_end = Rotation.from_quat([end_ori[1], end_ori[2], end_ori[3], end_ori[0]])

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

    def generate_dual_arm_trajectory(self) -> None:
        """Generate coordinated dual-arm handoff trajectory.

        15-phase sequence:
        0. Right waits, left moves to prepick above cube
        1. Left descends to prepick
        2. Left descends to pick
        3. Left grasps cube
        4. Left lifts with cube
        5. Both arms move to handoff positions
        6. Right approaches to receive
        7. Right grasps cube
        8. Left releases cube
        9. Left retreats
        10. Right lifts with cube
        11. Right moves to pre-place
        12. Right descends to place
        13. Right releases cube
        14. Both arms return home
        """
        cube_pos, _ = self.get_cube_pose()

        assert self.left_robot is not None
        assert self.right_robot is not None
        # Left arm key positions
        left_ee_pos, _ = self.left_robot.get_ee_pose()
        left_prepick = cube_pos + np.array([0.0, 0.0, self.clearance_height])
        left_pick = cube_pos + self.approach_offset
        left_handoff = self.handoff_position + np.array([0.0, self.handoff_offset, 0.0])
        left_retreat = self.left_home

        # Right arm key positions
        right_ee_pos, _ = self.right_robot.get_ee_pose()
        right_wait = np.array([0.0, -0.20, 0.25])
        right_pre_handoff = self.handoff_position + np.array([0.0, -0.12, 0.0])
        right_handoff = self.handoff_position + np.array([0.0, -self.handoff_offset, 0.0])
        right_lifted = right_handoff + np.array([0.0, 0.0, 0.05])
        right_preplace = self.target_position + np.array([0.0, 0.0, self.clearance_height])
        right_place = self.target_position + self.approach_offset

        # Left arm trajectory keyframes (16 frames for 15 phases)
        left_key_frames = [
            left_ee_pos,  # Start
            left_prepick,  # Phase 0 end: above cube
            left_prepick,  # Phase 1 end: stay at prepick
            left_pick,  # Phase 2 end: at pick
            left_pick,  # Phase 3 end: still at pick (grasping)
            left_prepick,  # Phase 4 end: lifted
            left_handoff,  # Phase 5 end: at handoff
            left_handoff,  # Phase 6 end: stay at handoff
            left_handoff,  # Phase 7 end: stay at handoff
            left_handoff,  # Phase 8 end: stay at handoff (releasing)
            left_retreat,  # Phase 9 end: retreated
            left_retreat,  # Phase 10-14: stay home
            left_retreat,
            left_retreat,
            left_retreat,
            left_retreat,
        ]

        left_orientations = [
            LEFT_ARM_DOWNWARD_ORIENTATION,  # Start
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 0
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 1
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 2
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 3
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 4
            LEFT_ARM_HANDOFF_ORIENTATION,  # 5 - rotate for handoff
            LEFT_ARM_HANDOFF_ORIENTATION,  # 6
            LEFT_ARM_HANDOFF_ORIENTATION,  # 7
            LEFT_ARM_HANDOFF_ORIENTATION,  # 8
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 9 - retreat
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 10
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 11
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 12
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 13
            LEFT_ARM_DOWNWARD_ORIENTATION,  # 14
        ]

        # Right arm trajectory keyframes
        right_key_frames = [
            right_ee_pos,  # Start
            right_wait,  # Phase 0 end: waiting
            right_wait,  # Phase 1 end
            right_wait,  # Phase 2 end
            right_wait,  # Phase 3 end
            right_wait,  # Phase 4 end
            right_pre_handoff,  # Phase 5 end: pre-handoff
            right_handoff,  # Phase 6 end: at handoff
            right_handoff,  # Phase 7 end: grasping
            right_handoff,  # Phase 8 end: wait for release
            right_handoff,  # Phase 9 end
            right_lifted,  # Phase 10 end: lifted
            right_preplace,  # Phase 11 end: pre-place
            right_place,  # Phase 12 end: at place
            right_place,  # Phase 13 end: releasing
            self.right_home,  # Phase 14 end: home
        ]

        right_orientations = [
            RIGHT_ARM_DOWNWARD_ORIENTATION,  # Start
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 0 - rotate to handoff
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 1
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 2
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 3
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 4
            RIGHT_ARM_RECEIVE_ORIENTATION,  # 5 - receive orientation
            RIGHT_ARM_RECEIVE_ORIENTATION,  # 6
            RIGHT_ARM_RECEIVE_ORIENTATION,  # 7
            RIGHT_ARM_RECEIVE_ORIENTATION,  # 8
            RIGHT_ARM_RECEIVE_ORIENTATION,  # 9
            RIGHT_ARM_HANDOFF_ORIENTATION,  # 10 - rotate while lifting
            RIGHT_ARM_DOWNWARD_ORIENTATION,  # 11 - downward for place
            RIGHT_ARM_DOWNWARD_ORIENTATION,  # 12
            RIGHT_ARM_DOWNWARD_ORIENTATION,  # 13
            RIGHT_ARM_DOWNWARD_ORIENTATION,  # 14
        ]

        self.left_trajectory = self.make_trajectory(
            left_key_frames, left_orientations, self.events_dt
        )
        self.right_trajectory = self.make_trajectory(
            right_key_frames, right_orientations, self.events_dt
        )
        self.trajectory_index = 0


def main():
    """Main execution function."""
    print("=" * 70)
    print("Stationary AI Dual-Arm Pick-and-Place Demo (MuJoCo)")
    print("=" * 70)
    print()
    print("The left arm picks up the cube and hands it off to the right arm,")
    print("which then places it at the target location.")
    print()
    print("15-phase sequence:")
    print("  Phases 0-4: Left arm picks up cube")
    print("  Phases 5-9: Handoff from left to right arm")
    print("  Phases 10-14: Right arm places cube")
    print()
    print("=" * 70)
    print()

    pick_place = StationaryAIPickPlace()
    pick_place.setup_scene()
    pick_place.reset()

    task_completed = False

    with mujoco.viewer.launch_passive(pick_place.model, pick_place.data) as viewer:
        # Set viewer camera for better initial view
        viewer.cam.lookat[:] = [0.0, 0.0, 0.15]
        viewer.cam.distance = 1.2
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        dt = pick_place.model.opt.timestep

        while viewer.is_running():
            step_start = time.time()

            if not task_completed:
                pick_place.forward()

            if pick_place.is_done() and not task_completed:
                print("\nTask complete!")
                print("\nDual-arm handoff pick-and-place finished successfully.")
                task_completed = True

            mujoco.mj_step(pick_place.model, pick_place.data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\nStationary AI pick-and-place demo completed")


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
