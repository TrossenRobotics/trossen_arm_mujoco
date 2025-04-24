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

from dm_control.mujoco.engine import Physics
from dm_control.suite import base
import numpy as np

from trossen_arm_mujoco.constants import START_ARM_POSE
from trossen_arm_mujoco.utils import (
    get_observation_base,
    make_sim_env,
    plot_observation_images,
    sample_box_pose,
    set_observation_images,
)


class TrossenAIStationaryEETask(base.Task):
    """
    A base task for bimanual Cartesian manipulation with Trossen AI robotic arms in the Trossen AI
    Stationary Kit form factor.

    :param random: Randomization seed for environment initialization, defaults to ``None``.
    :param onscreen_render: Whether to enable on-screen rendering, defaults to ``False``.
    :param cam_list: List of cameras for observation capture, defaults to ``[]``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render=False,
        cam_list: list[str] = [],
    ):
        super().__init__(random=random)
        self.cam_list = cam_list
        if self.cam_list == []:
            self.cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]

    def before_step(self, action: np.ndarray, physics: Physics) -> None:
        """
        Apply the action to the robotic arms before stepping the simulation.

        :param action: The action vector containing position and gripper commands.
        :param physics: The simulation physics instance.
        """
        a_len = len(action) // 2
        action_left = action[:a_len]
        action_right = action[a_len:]

        # set mocap position and quat
        # left
        np.copyto(physics.data.mocap_pos[0], action_left[:3])
        np.copyto(physics.data.mocap_quat[0], action_left[3:7])
        # right
        np.copyto(physics.data.mocap_pos[1], action_right[:3])
        np.copyto(physics.data.mocap_quat[1], action_right[3:7])

        physics.data.qpos[6] = action_left[7]
        physics.data.qpos[7] = action_left[7]
        physics.data.qpos[14] = action_right[7]
        physics.data.qpos[15] = action_right[7]

    def initialize_robots(self, physics: Physics) -> None:
        """
        Initialize the robots by resetting joint positions and aligning mocap bodies with end-effectors.

        :param physics: The simulation physics engine.
        """
        # reset joint position
        physics.named.data.qpos[:12] = START_ARM_POSE[:6] + START_ARM_POSE[8:14]

        # reset mocap to align with end effector
        np.copyto(physics.data.mocap_pos[0], [-0.31718881+0.142+0.0115, 0.0, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881-0.142-0.0115, 0.0, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])

    def initialize_episode(self, physics: Physics):
        """
        Set up the environment state at the beginning of each episode.

        :param physics: The simulation physics engine.
        """
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieve the environment state.

        :param physics: The simulation physics engine.
        :raises NotImplementedError: This function must be implemented in derived classes.
        """
        raise NotImplementedError

    def get_position(self, physics: Physics) -> np.ndarray:
        """
        Get the current joint positions of the robot.

        :param physics: The simulation physics engine.
        :return: The joint positions.
        """
        positions = physics.data.qpos.copy()
        return positions[:16]

    def get_velocity(self, physics: Physics) -> np.ndarray:
        """
        Get the current joint velocities of the robot.

        :param physics: The simulation physics engine.
        :return: The joint velocities.
        """
        velocities = physics.data.qvel.copy()
        return velocities[:16]

    def get_observation(self, physics: Physics) -> dict:
        """
        Retrieve the robot's observation data, including joint positions, velocities, and camera images.

        :param physics: The simulation physics engine.
        :return: The current observation state.
        """
        obs = get_observation_base(physics, self.cam_list)
        obs["qpos"] = self.get_position(physics)
        obs["qvel"] = self.get_velocity(physics)
        obs["env_state"] = self.get_env_state(physics)
        obs["mocap_pose_left"] = np.concatenate(
            [physics.data.mocap_pos[0], physics.data.mocap_quat[0]]
        ).copy()
        obs["mocap_pose_right"] = np.concatenate(
            [physics.data.mocap_pos[1], physics.data.mocap_quat[1]]
        ).copy()
        obs["gripper_ctrl"] = physics.data.ctrl.copy()

        pos = physics.named.data.xpos["mocap_right"]
        print(f"[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")

        
        return obs

    def get_reward(self, physics: Physics) -> int:
        """
        Compute the task-specific reward.

        :param physics: The simulation physics engine.
        :raises NotImplementedError: This function must be implemented in derived classes.
        """
        raise NotImplementedError


class TransferCubeEETask(TrossenAIStationaryEETask):
    """
    A task where a cube must be transferred between two robotic arms.

    :param random: Random seed for environment variability, defaults to ``None``.
    :param onscreen_render: Whether to enable real-time rendering, defaults to ``False``.
    :param cam_list: List of cameras to capture observations, defaults to ``None``.
    """

    def __init__(
        self,
        random: int | None = None,
        onscreen_render: bool = False,
        cam_list: list[str] = [],
    ):
        super().__init__(
            random=random,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
        self.max_reward = 4

    def initialize_episode(self, physics: Physics) -> None:
        """
        Set up the simulation environment at the start of an episode.

        :param physics: The simulation physics engine.
        """
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id("red_box_joint", "joint")
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics: Physics) -> np.ndarray:
        """
        Retrieve the environment state specific to this task.

        :param physics: The simulation physics engine.
        :return: The state of the environment.
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics: Physics) -> int:
        """
        Compute the reward based on the cube's interaction with the robot and the environment.

        :param physics: The simulation physics engine.
        :return: The computed reward.
        """
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, "geom")
            name_geom_2 = physics.model.id2name(id_geom_2, "geom")
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_left_gripper = (
            "red_box",
            "left/gripper_follower_left",
        ) in all_contact_pairs
        touch_right_gripper = (
            "red_box",
            "right/gripper_follower_left",
        ) in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table:  # lifted
            reward = 2
        if touch_left_gripper:  # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table:  # successful transfer
            reward = 4
        return reward


def test_ee_sim_env():
    onscreen_render = True
    cam_list = ["cam_high", "cam_low", "cam_left_wrist", "cam_right_wrist"]
    env = make_sim_env(
        TransferCubeEETask,
        task_name="sim_transfer_cube",
        onscreen_render=onscreen_render,
        cam_list=cam_list,
    )
    ts = env.reset()
    episode = [ts]
    # setup plotting
    if onscreen_render:
        plt_imgs = plot_observation_images(ts.observation, cam_list)
    for t in range(1000):
        action = np.random.uniform(-0.1, 0.1, 23)
        ts = env.step(action)
        episode.append(ts)
        if onscreen_render:
            plt_imgs = set_observation_images(ts.observation, plt_imgs, cam_list)


if __name__ == "__main__":
    test_ee_sim_env()
