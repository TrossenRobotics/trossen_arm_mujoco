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
"""WidowX AI Pick-and-Place Demonstration.

Demonstrates pick-and-place manipulation using the WidowX AI robot.

Usage:
    python trossen_arm_mujoco/scripts/wxai_pick_place.py
"""

from __future__ import annotations

import os
import sys

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.ik_controller import IKController

# Default configuration constants
DEFAULT_CUBE_SIZE = np.array([0.05, 0.05, 0.05])
DEFAULT_CUBE_POSITION = np.array([0.35, -0.15, 0.025])
DEFAULT_CUBE_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])
DEFAULT_TARGET_POSITION = np.array([0.35, 0.15, 0.025])
DEFAULT_HOME_POSITION = np.array([0.2, 0.0, 0.3])
DEFAULT_EVENTS_DT = [1200, 800, 200, 800, 1200, 800, 200, 800, 1200]

# Trajectory parameters
CLEARANCE_HEIGHT = 0.15
APPROACH_OFFSET = np.array([0.0, 0.0, 0.0])
DOWNWARD_ORIENTATION = np.array([0.70710678, 0.0, 0.70710678, 0.0])

# Scene configuration
SCENE_XML_PATH = "trossen_arm_mujoco/assets/wxai/scene_wxai_pick_place.xml"

# Robot controller configuration
ARM_JOINT_NAMES = [f"joint_{i}" for i in range(6)]
GRIPPER_JOINT_NAMES = ["left_carriage_joint"]
DEFAULT_DOF_POSITIONS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044])

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03
POSITION_THRESHOLD = 0.04
MAX_STEPS_PER_WAYPOINT = 100


class WXAIPickPlace:
    """Pick-and-place task with trajectory-based motion control."""

    def __init__(
        self,
        events_dt: list[int] | None = None,
        cube_initial_position: np.ndarray | None = None,
        cube_initial_orientation: np.ndarray | None = None,
        target_position: np.ndarray | None = None,
    ):
        """Initialize pick-and-place task.

        Args:
            events_dt: List of time deltas for events in the task sequence.
            cube_initial_position: Initial position [x, y, z] of the cube in meters.
            cube_initial_orientation: Initial orientation quaternion [w, x, y, z] of the cube.
            target_position: Target position [x, y, z] for placing the cube in meters.
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
            target_position
            if target_position is not None
            else DEFAULT_TARGET_POSITION.copy()
        )

        self.events_dt = (
            events_dt if events_dt is not None else DEFAULT_EVENTS_DT.copy()
        )

        self.clearance_height = CLEARANCE_HEIGHT
        self.approach_offset = APPROACH_OFFSET.copy()
        self.home_position = DEFAULT_HOME_POSITION.copy()

        self.model = None
        self.data = None
        self.robot = None
        self.cube_body_id = None
        self.trajectory = None
        self.trajectory_index = 0
        self.waypoint_step_count = 0

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, cube, and environment."""
        # Load MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)

        # Initialize robot controller
        self.robot = IKController(
            model=self.model,
            data=self.data,
            robot_type="wxai",
            arm_joint_names=ARM_JOINT_NAMES,
            gripper_joint_names=GRIPPER_JOINT_NAMES,
            ik_scale=IK_SCALE,
            ik_damping=IK_DAMPING,
        )

        # Get cube body ID
        self.cube_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "cube"
        )

    def forward(self) -> bool:
        """Execute one simulation step of the pick-and-place trajectory.

        Returns:
            bool: True if trajectory is in progress, False if complete.
        """
        if self.is_done():
            return False

        if self.trajectory is None:
            self.generate_pick_place_trajectory()

        assert self.trajectory is not None
        assert self.robot is not None

        if self.trajectory_index < len(self.trajectory):
            goal_position, goal_orientation, _ = self.trajectory[self.trajectory_index]

            # Apply one IK step toward goal
            error = self.robot.set_ee_pose(
                target_position=goal_position,
                target_orientation=goal_orientation,
                position_only=False,
            )

            # Increment waypoint step counter
            self.waypoint_step_count += 1

            # Advance waypoint if reached threshold OR timeout
            if (
                error < POSITION_THRESHOLD
                or self.waypoint_step_count >= MAX_STEPS_PER_WAYPOINT
            ):
                self.trajectory_index += 1
                self.waypoint_step_count = 0  # Reset counter for next waypoint

                assert self.robot is not None
                # Calculate phase boundaries
                phase_boundaries = [0]
                cumulative = 0
                for duration in self.events_dt:
                    cumulative += duration
                    phase_boundaries.append(cumulative)

                # Close gripper during pick phase (phase 2: indices phase_boundaries[2] to phase_boundaries[3])
                if phase_boundaries[2] <= self.trajectory_index < phase_boundaries[3]:
                    self.robot.close_gripper()
                # Open gripper during place phase (phase 6: indices phase_boundaries[6] to phase_boundaries[7])
                elif phase_boundaries[6] <= self.trajectory_index < phase_boundaries[7]:
                    self.robot.open_gripper()

        return True

    def is_done(self) -> bool:
        """Check if pick-and-place task is complete.

        Returns:
            bool: True if all trajectory waypoints have been executed.
        """
        return self.trajectory is not None and self.trajectory_index >= len(
            self.trajectory
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
        """Reset robot to default pose and clear trajectory."""
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")

        # Reset robot to default pose
        for i, joint_name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = DEFAULT_DOF_POSITIONS[i]

        # Reset gripper to open
        self.robot.open_gripper()

        self.trajectory = None
        self.trajectory_index = 0

        # Forward kinematics
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

        # Find the freejoint for the cube body
        cube_joint_id = None
        for i in range(self.model.njnt):
            body_id = self.model.jnt_bodyid[i]
            if body_id == self.cube_body_id:
                cube_joint_id = i
                break

        if cube_joint_id is None:
            raise RuntimeError("Could not find joint for cube body")

        # Get qpos address for the cube's freejoint (7 degrees_of_freedom: 3 pos + 4 quat)
        qpos_addr = self.model.jnt_qposadr[cube_joint_id]

        # Set position (x, y, z)
        self.data.qpos[qpos_addr : qpos_addr + 3] = reset_position

        # Set orientation (quaternion w, x, y, z)
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = reset_orientation

        # Zero out velocities
        qvel_addr = self.model.jnt_dofadr[cube_joint_id]
        self.data.qvel[qvel_addr : qvel_addr + 6] = 0.0

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)

    def get_cube_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Get current cube position and orientation.

        Returns:
            position: Cube position [x, y, z]
            orientation: Cube orientation quaternion [w, x, y, z]
        """
        assert self.data is not None
        assert self.cube_body_id is not None
        # Get cube position
        cube_pos = self.data.xpos[self.cube_body_id].copy()

        # Get cube orientation (convert rotation matrix to quaternion)
        cube_xmat = self.data.xmat[self.cube_body_id].reshape(3, 3)
        rotation = Rotation.from_matrix(cube_xmat)
        quat_xyzw = rotation.as_quat()  # scipy format: [x, y, z, w]
        cube_quat = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
        )  # [w, x, y, z]

        return cube_pos, cube_quat

    def make_trajectory(
        self,
        key_frames: list[np.ndarray],
        orientations: list[np.ndarray],
        dt: list[int],
    ) -> list[tuple[np.ndarray, np.ndarray, int]]:
        """Generate smooth trajectory via linear interpolation between keyframes.

        Args:
            key_frames: Position waypoints [x, y, z] in meters. Length must be len(dt) + 1.
            orientations: Orientation quaternions [w, x, y, z] for each keyframe.
            dt: Duration in steps for each trajectory segment.

        Returns:
            List of (position, orientation, cumulative_step) tuples.

        Raises:
            ValueError: If array lengths are incompatible.
        """
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

            # Convert quaternions to Rotation objects for SLERP
            rot_start = Rotation.from_quat(
                [start_ori[1], start_ori[2], start_ori[3], start_ori[0]]
            )  # [x,y,z,w]
            rot_end = Rotation.from_quat(
                [end_ori[1], end_ori[2], end_ori[3], end_ori[0]]
            )  # [x,y,z,w]

            # Linear interpolation for position, SLERP for orientation
            for step in range(n_steps):
                alpha = step / n_steps if n_steps > 0 else 0.0
                interpolated_pos = start_pos + alpha * (end_pos - start_pos)

                # SLERP for smooth orientation interpolation
                if alpha == 0.0:
                    interpolated_ori = start_ori
                else:
                    # Normalized linear interpolation for quaternions
                    q = rot_start.as_quat() * (1 - alpha) + rot_end.as_quat() * alpha
                    norm = np.linalg.norm(q)
                    if norm > 0.0:
                        q = q / norm
                    rot_interp = Rotation.from_quat(q)
                    # Convert back to [w,x,y,z]
                    quat_xyzw = rot_interp.as_quat()
                    interpolated_ori = np.array(
                        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]
                    )

                trajectory.append(
                    (interpolated_pos, interpolated_ori, cumulative_step + step)
                )

            cumulative_step += n_steps

        # Add final keyframe
        trajectory.append(
            (
                np.array(key_frames[-1], dtype=np.float64),
                np.array(orientations[-1], dtype=np.float64),
                cumulative_step,
            )
        )

        return trajectory

    def generate_pick_place_trajectory(self) -> None:
        """Generate complete pick-and-place trajectory from current state.

        Creates a 9-phase trajectory with smooth linear interpolation:
        1. Move to pre-pick position above cube
        2. Descend to pick approach height
        3. Close gripper
        4. Lift cube with clearance
        5. Move to pre-place position above target
        6. Descend to place approach height
        7. Open gripper
        8. Retreat with clearance
        9. Return to home position
        """
        cube_pos, _ = self.get_cube_pose()
        assert self.robot is not None
        current_ee_pos, _ = self.robot.get_ee_pose()

        key_frames = [
            current_ee_pos,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            cube_pos + self.approach_offset,
            cube_pos + self.approach_offset,
            cube_pos + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.target_position + self.approach_offset,
            self.target_position + self.approach_offset,
            self.target_position + np.array([0.0, 0.0, self.clearance_height]),
            self.home_position,
        ]

        goal_orientation = DOWNWARD_ORIENTATION
        orientations = [goal_orientation for _ in key_frames]

        self.trajectory = self.make_trajectory(key_frames, orientations, self.events_dt)
        self.trajectory_index = 0


def main():
    """Main execution function matching IsaacSim structure."""
    print("=" * 70)
    print("WidowX AI Pick-and-Place Demo (MuJoCo)")
    print("=" * 70)
    print()
    print("The robot will pick up the blue cube and place it at the target location.")
    print()
    print("Task sequence:")
    print("  0. Move to pre-pick position above cube")
    print("  1. Rotate to downward orientation at pre-pick")
    print("  2. Descend to pick approach height")
    print("  3. Close gripper")
    print("  4. Lift cube with clearance")
    print("  5. Move to pre-place position above target")
    print("  6. Descend to place approach height")
    print("  7. Open gripper")
    print("  8. Retreat with clearance")
    print("  9. Return to home position")
    print()
    print("=" * 70)
    print()

    # Create and setup the pick-and-place task
    pick_place = WXAIPickPlace()
    pick_place.setup_scene()

    # Reset to initial state
    pick_place.reset()

    task_completed = False

    # Launch viewer and run simulation with real-time synchronization
    import time

    with mujoco.viewer.launch_passive(pick_place.model, pick_place.data) as viewer:
        # Set viewer to full screen
        viewer.cam.lookat[:] = [0.3, 0.0, 0.15]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

        # Set simulation timestep for real-time sync
        dt = pick_place.model.opt.timestep  # Should be 1/60 = 0.0166... seconds

        while viewer.is_running():
            step_start = time.time()

            # Execute pick-and-place trajectory
            if not task_completed:
                pick_place.forward()

            # Check if task is done
            if pick_place.is_done() and not task_completed:
                print("\n✓ Task complete!")
                print("\nYou can manually reset the cube or close the viewer.")
                task_completed = True

            # Step simulation
            mujoco.mj_step(pick_place.model, pick_place.data)

            # Sync viewer
            viewer.sync()

            # Real-time synchronization: sleep to match wall clock time
            elapsed = time.time() - step_start
            if elapsed < dt:
                time.sleep(dt - elapsed)

    print("\n✓ Pick-and-place demo completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping pick-and-place demo...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
