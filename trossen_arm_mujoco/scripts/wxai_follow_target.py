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
"""WidowX AI Follow Target Demonstration.

Demonstrates continuous end effector tracking of a movable target cube.

Usage:
    python trossen_arm_mujoco/scripts/wxai_follow_target.py
"""

from __future__ import annotations

import os
import sys

import mujoco
import mujoco.viewer
import numpy as np

# Add parent directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from scripts.ik_controller import IKController

# Default target configuration
DEFAULT_TARGET_POSITION = np.array([0.3, 0.0, 0.2])
DEFAULT_TARGET_ORIENTATION = np.array([1.0, 0.0, 0.0, 0.0])

# Scene configuration
SCENE_XML_PATH = "trossen_arm_mujoco/assets/wxai/scene_wxai_follow_target.xml"

# Robot controller configuration
ARM_JOINT_NAMES = [f"joint_{i}" for i in range(6)]
GRIPPER_JOINT_NAMES = ["left_carriage_joint"]
DEFAULT_DOF_POSITIONS = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.044])

# IK configuration
IK_SCALE = 1.0
IK_DAMPING = 0.03


class WXAIFollowTarget:
    """Real-time target tracking demonstration.

    Continuously tracks a movable target cube's pose, commanding the robot's
    end effector to match the target's position and orientation.
    """

    def __init__(
        self,
        target_initial_position: np.ndarray | None = None,
        target_initial_orientation: np.ndarray | None = None,
    ):
        """Initialize target tracking task."""
        self.target_initial_position = (
            target_initial_position
            if target_initial_position is not None
            else DEFAULT_TARGET_POSITION.copy()
        )
        self.target_initial_orientation = (
            target_initial_orientation
            if target_initial_orientation is not None
            else DEFAULT_TARGET_ORIENTATION.copy()
        )

        self.model = None
        self.data = None
        self.robot = None
        self.target_body_id = None

    def setup_scene(self) -> None:
        """Initialize simulation scene with robot, target cube, and environment."""
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

        # Get target body ID
        self.target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target_cube"
        )

        # Get mocap body ID - mocap bodies are indexed separately from regular bodies
        # In MuJoCo, mocap_pos and mocap_quat are indexed by mocap body index (0-based)
        # Our target_cube is the only mocap body, so it's at index 0
        self.target_mocap_id = 0

        assert self.data is not None
        # Set target initial position using mocap
        self.data.mocap_pos[self.target_mocap_id] = self.target_initial_position
        self.data.mocap_quat[self.target_mocap_id] = self.target_initial_orientation

    def forward(self) -> None:
        """Execute one tracking step by commanding end effector to current target pose."""
        assert self.data is not None
        assert self.robot is not None
        # Get current target pose from mocap body
        target_position = self.data.mocap_pos[self.target_mocap_id].copy()
        target_orientation = self.data.mocap_quat[self.target_mocap_id].copy()

        # Command robot to track target (full pose: position AND orientation)
        self.robot.set_ee_pose(
            target_position=target_position,
            target_orientation=target_orientation,
            position_only=False,
        )

    def reset(
        self,
        target_position: np.ndarray | None = None,
        target_orientation: np.ndarray | None = None,
    ) -> None:
        """Reset robot and target to initial state."""
        if self.robot is None:
            raise RuntimeError("Cannot reset robot: robot not initialized.")
        if self.target_body_id is None:
            raise RuntimeError("Cannot reset target: target not initialized.")

        # Reset robot to default pose
        for i, joint_name in enumerate(ARM_JOINT_NAMES):
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = DEFAULT_DOF_POSITIONS[i]

        # Reset gripper
        for joint_name in GRIPPER_JOINT_NAMES:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            qpos_addr = self.model.jnt_qposadr[joint_id]
            self.data.qpos[qpos_addr] = DEFAULT_DOF_POSITIONS[6]

        # Reset target to initial or specified pose
        reset_position = (
            target_position
            if target_position is not None
            else self.target_initial_position
        )
        reset_orientation = (
            target_orientation
            if target_orientation is not None
            else self.target_initial_orientation
        )
        self.data.mocap_pos[self.target_mocap_id] = reset_position
        self.data.mocap_quat[self.target_mocap_id] = reset_orientation

        # Forward kinematics
        mujoco.mj_forward(self.model, self.data)


def main():
    """Main execution function matching IsaacSim structure."""
    print("=" * 70)
    print("WidowX AI Follow Target Demo (MuJoCo)")
    print("=" * 70)
    print()
    print("The robot will continuously track the blue target cube.")
    print("You can manually move the cube in the viewer to test tracking.")
    print()
    print("Controls:")
    print("  - Double-click the target cube to select it")
    print("  - Ctrl + Right-click drag to move the cube")
    print("  - Press ESC to exit")
    print()
    print("=" * 70)
    print()

    # Create and setup the follow target task
    follow_target = WXAIFollowTarget()
    follow_target.setup_scene()

    # Reset to initial state
    follow_target.reset()

    # Launch viewer and run simulation
    with mujoco.viewer.launch_passive(
        follow_target.model, follow_target.data
    ) as viewer:
        while viewer.is_running():
            # Execute tracking step
            follow_target.forward()

            # Step simulation
            mujoco.mj_step(follow_target.model, follow_target.data)

            # Sync viewer
            viewer.sync()

    print("\n✓ Follow target demo completed")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping follow target demo...")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
