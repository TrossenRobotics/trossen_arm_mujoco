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
"""Stationary AI Transfer Cube Environment.

Shared simulation environment for recording demonstrations and evaluating policies,
following the ACT/ALOHA conventions (https://arxiv.org/abs/2304.13705):

- Control at 50 Hz (each step runs 10 physics steps of 0.002 s).
- Action: 14 absolute joint targets, 7 per arm (6 joints + gripper), left arm first.
- Observation: qpos and qvel of the same 14 joints, plus 480x640 RGB images.
- Reward: staged 0-4, success when the cube is placed on the target.

Task: the left arm picks the cube, hands it over to the right arm, and the right arm
places it on the target.
"""

from __future__ import annotations

import os

import mujoco
import numpy as np

from trossen_arm_mujoco import ASSETS_DIR
from trossen_arm_mujoco.controller import GRIPPER_OPEN_POSITION

# Scene configuration
SCENE_XML_PATH = os.path.join(ASSETS_DIR, "stationary_ai", "scene_stationary_ai_pick_place.xml")

# Control configuration
DT = 0.02  # 50 Hz
CAMERA_NAMES = ["cam_high", "cam_low", "follower_left_camera", "follower_right_camera"]
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640

# Task configuration
CUBE_POSITION_LOW = np.array([-0.1, 0.15, 0.045])
CUBE_POSITION_HIGH = np.array([0.1, 0.3, 0.045])
TARGET_POSITION = np.array([0.0, -0.25, 0.045])
TARGET_TOLERANCE = 0.03  # meters, in the xy plane
MAX_REWARD = 4


class TransferCubeEnv:
    """Dual-arm cube transfer environment for Stationary AI robot."""

    def __init__(self, camera_names: list[str] | None = None):
        """Initialize environment.

        :param camera_names: Cameras to render in observations (default: all four).
        """
        self.camera_names = camera_names if camera_names is not None else CAMERA_NAMES

        self.model = mujoco.MjModel.from_xml_path(SCENE_XML_PATH)
        self.data = mujoco.MjData(self.model)
        self.renderer = mujoco.Renderer(self.model, IMAGE_HEIGHT, IMAGE_WIDTH)
        self.n_substeps = round(DT / self.model.opt.timestep)
        self.rng = np.random.default_rng()

        # Joints driven by the 14 actuators
        actuated_joint_ids = self.model.actuator_trnid[:, 0]
        self.qpos_indices = self.model.jnt_qposadr[actuated_joint_ids]
        self.qvel_indices = self.model.jnt_dofadr[actuated_joint_ids]
        self.gripper_actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in ["follower_left_gripper", "follower_right_gripper"]
        ]
        # Mimic finger of each gripper
        self.mimic_finger_qpos_indices = [
            self.model.jnt_qposadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            for name in ["follower_left_right_carriage_joint", "follower_right_right_carriage_joint"]
        ]

        # Cube and contact bodies for the reward
        self.cube_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        cube_joint_id = self.model.body_jntadr[self.cube_body_id]
        self.cube_qpos_addr = self.model.jnt_qposadr[cube_joint_id]
        self.table_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "tabletop_link")
        self.left_finger_body_ids = self._body_ids("follower_left_carriage_left", "follower_left_carriage_right")
        self.right_finger_body_ids = self._body_ids("follower_right_carriage_left", "follower_right_carriage_right")

    def _body_ids(self, *names: str) -> set[int]:
        return {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name) for name in names}

    def reset(self, cube_position: np.ndarray | None = None, seed: int | None = None) -> dict:
        """Reset arms to home with grippers open and place the cube.

        :param cube_position: Cube position [x, y, z]. Random in the task region if None.
        :param seed: Re-seed the random generator (only needed on the first reset).
        :return: Initial observation.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        if cube_position is None:
            cube_position = self.rng.uniform(CUBE_POSITION_LOW, CUBE_POSITION_HIGH)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.cube_qpos_addr : self.cube_qpos_addr + 3] = cube_position
        self.data.qpos[self.cube_qpos_addr + 3 : self.cube_qpos_addr + 7] = [1.0, 0.0, 0.0, 0.0]
        self.data.ctrl[self.gripper_actuator_ids] = GRIPPER_OPEN_POSITION
        self.data.qpos[self.qpos_indices] = self.data.ctrl
        # mj_forward does not enforce the mimic constraint, so open the mimic fingers
        self.data.qpos[self.mimic_finger_qpos_indices] = GRIPPER_OPEN_POSITION
        mujoco.mj_forward(self.model, self.data)

        return self.get_observation()

    def step(self, action: np.ndarray) -> tuple[dict, int]:
        """Apply 14 joint targets and simulate one control step.

        :param action: Joint targets [left 6 joints, left gripper, right 6 joints, right gripper].
        :return: Tuple of (observation, reward).
        """
        self.data.ctrl[:] = action
        mujoco.mj_step(self.model, self.data, nstep=self.n_substeps)
        return self.get_observation(), self.get_reward()

    def close(self) -> None:
        """Release the renderer."""
        self.renderer.close()

    def get_observation(self) -> dict:
        """Get joint positions, joint velocities and camera images."""
        images = {}
        for camera_name in self.camera_names:
            self.renderer.update_scene(self.data, camera=camera_name)
            images[camera_name] = self.renderer.render()

        return {
            "qpos": self.data.qpos[self.qpos_indices].copy(),
            "qvel": self.data.qvel[self.qvel_indices].copy(),
            "images": images,
        }

    def get_reward(self) -> int:
        """Staged reward.

        1: left gripper touches the cube
        2: left gripper lifts the cube off the table
        3: right gripper holds the cube without the left gripper (handover)
        4: cube rests on the target and is released
        """
        touching = set()
        for contact in self.data.contact[: self.data.ncon]:
            body1 = self.model.geom_bodyid[contact.geom1]
            body2 = self.model.geom_bodyid[contact.geom2]
            if body1 == self.cube_body_id:
                touching.add(body2)
            elif body2 == self.cube_body_id:
                touching.add(body1)

        touch_left = bool(touching & self.left_finger_body_ids)
        touch_right = bool(touching & self.right_finger_body_ids)
        touch_table = self.table_body_id in touching

        cube_position = self.data.xpos[self.cube_body_id]
        on_target = np.linalg.norm(cube_position[:2] - TARGET_POSITION[:2]) < TARGET_TOLERANCE

        if on_target and touch_table and not touch_left and not touch_right:
            return 4
        if touch_right and not touch_left and not touch_table:
            return 3
        if touch_left and not touch_table:
            return 2
        if touch_left:
            return 1
        return 0
