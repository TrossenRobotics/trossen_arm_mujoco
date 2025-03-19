import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import numpy as np
import matplotlib.pyplot as plt
from utils import sample_box_pose, get_observation_base, make_sim_env, plot_observation_images, set_observation_images
from constants import DT, START_ARM_POSE, XML_DIR, BOX_POSE

class TrossenAIBimanualEETask(base.Task):
    """
    Base class for bimanual robotic manipulation tasks in the Trossen AI simulation.

    :param random: Randomization seed for environment initialization, defaults to ``None``.
    :type random: int, optional
    :param onscreen_render: Whether to enable on-screen rendering, defaults to ``False``.
    :type onscreen_render: bool, optional
    :param camera_list: List of cameras for observation capture, defaults to ``None``.
    :type camera_list: list, optional
    """
    def __init__(self, random=None, onscreen_render=False, camera_list=None):
        super().__init__(random=random)
        self.camera_list = camera_list if camera_list else ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]

    def before_step(self, action, physics):
        """
        Apply the action to the robotic arms before stepping the simulation.

        :param action: The action vector containing position and gripper commands.
        :type action: np.ndarray
        :param physics: The simulation physics instance.
        :type physics: mujoco.Physics
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

        # # set gripper
        # g_left_ctrl = FOLLOWER_GRIPPER_POSITION_UNNORMALIZE_FN(action_left[7])
        # g_right_ctrl = FOLLOWER_GRIPPER_POSITION_UNNORMALIZE_FN(action_right[7])
        # np.copyto(
        #     physics.data.ctrl,
        #     np.array([g_left_ctrl, -g_left_ctrl, g_right_ctrl, -g_right_ctrl])
        # )

        physics.data.qpos[6] = action_left[7]
        physics.data.qpos[7] = action_left[7]
        physics.data.qpos[14] = action_right[7]
        physics.data.qpos[15] = action_right[7]


    def initialize_robots(self, physics):
        """
        Initialize the robots by resetting joint positions and aligning mocap bodies with end-effectors.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        """
        # reset joint position
        physics.named.data.qpos[:12] = START_ARM_POSE[:6] + START_ARM_POSE[8:14]

        # reset mocap to align with end effector
        # to obtain these numbers:
        # (1) make an ee_sim env and reset to the same start_pose
        # (2) get env._physics.named.data.xpos['vx300s_left/gripper_link']
        #     get env._physics.named.data.xquat['vx300s_left/gripper_link']
        #     repeat the same for right side
        np.copyto(physics.data.mocap_pos[0], [-0.31718881, 0.0, 0.29525084])
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])
        # right
        np.copyto(physics.data.mocap_pos[1], np.array([0.31718881, 0.0, 0.29525084]))
        np.copyto(physics.data.mocap_quat[1],  [1, 0, 0, 0])

        # reset gripper control
        # close_gripper_control = np.array([
        #     FOLLOWER_GRIPPER_POSITION_CLOSE,
        #     -FOLLOWER_GRIPPER_POSITION_CLOSE,
        #     FOLLOWER_GRIPPER_POSITION_CLOSE,
        #     -FOLLOWER_GRIPPER_POSITION_CLOSE,
        # ])
        # np.copyto(physics.data.ctrl, close_gripper_control)

    def initialize_episode(self, physics):
        """
        Set up the environment state at the beginning of each episode.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        """
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """
        Retrieve the environment state.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :raises NotImplementedError: This function must be implemented in derived classes.
        """
        raise NotImplementedError

    def get_position(self, physics):
        """
        Get the current joint positions of the robot.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :return: The joint positions.
        :rtype: np.ndarray
        """
        positions = physics.data.qpos.copy()
        return positions[:16]

    def get_velocity(self, physics):
        """
        Get the current joint velocities of the robot.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :return: The joint velocities.
        :rtype: np.ndarray
        """
        velocities = physics.data.qvel.copy()
        return velocities[:16]

    def get_observation(self, physics):
        """
        Retrieve the robot's observation data, including joint positions, velocities, and camera images.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :return: The current observation state.
        :rtype: dict
        """
        obs = get_observation_base(physics, self.camera_list)
        obs['qpos'] = self.get_position(physics)
        obs['qvel'] = self.get_velocity(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['mocap_pose_left'] = np.concatenate([
            physics.data.mocap_pos[0],
            physics.data.mocap_quat[0]
        ]).copy()
        obs['mocap_pose_right'] = np.concatenate([
            physics.data.mocap_pos[1],
            physics.data.mocap_quat[1]
        ]).copy()
        # used when replaying joint trajectory
        obs['gripper_ctrl'] = physics.data.ctrl.copy()
        return obs

    def get_reward(self, physics):
        """
        Compute the task-specific reward.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :raises NotImplementedError: This function must be implemented in derived classes.
        """
        raise NotImplementedError


class TransferCubeEETask(TrossenAIBimanualEETask):
    def __init__(self, random=None, onscreen_render=False, camera_list=None):
        super().__init__(random=random, onscreen_render=onscreen_render, camera_list=camera_list)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """
        Set up the simulation environment at the start of an episode.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        """        
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)

        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        """
        Retrieve the environment state specific to this task.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :return: The state of the environment.
        :rtype: np.ndarray
        """
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
        """
        Compute the reward based on the cube's interaction with the robot and the environment.

        :param physics: The simulation physics engine.
        :type physics: mujoco.Physics
        :return: The computed reward.
        :rtype: int
        """
        # return whether left gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_left_gripper = ("red_box", "left/gripper_follower_left") in all_contact_pairs
        touch_right_gripper = ("red_box", "right/gripper_follower_left") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not touch_table: # lifted
            reward = 2
        if touch_left_gripper: # attempted transfer
            reward = 3
        if touch_left_gripper and not touch_table: # successful transfer
            reward = 4
        return reward


def test_ee_sim_env():
    onscreen_render = True
     # setup the environment
    camera_list = ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    env = make_sim_env(
        TransferCubeEETask, 
        task_name='sim_transfer_cube', 
        onscreen_render=onscreen_render, 
        camera_list = camera_list
    )
    # print(f"Action space: {env.action_spec().shape}")
    ts = env.reset()
    episode = [ts]
    # setup plotting
    if onscreen_render:
        plt_imgs = plot_observation_images(ts.observation, camera_list)
    for t in range(1000):
        action = np.random.uniform(-0.1, 0.1, 23)
        ts = env.step(action)
        episode.append(ts)
        if onscreen_render:
            plt_imgs = set_observation_images(ts.observation, plt_imgs, camera_list)

if __name__ == '__main__':
    test_ee_sim_env()
