import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np
from utils import sample_box_pose
from utils import make_sim_env, get_observation_base, plot_observation_images

XML_DIR="assets"
DT = 0.02
BOX_POSE = [None] # to be changed from outside
START_ARM_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ,0.0])

class BimanualViperXTask(base.Task):
    def __init__(self, random=None, onscreen_render=False, camera_list=None):
        super().__init__(random=random)
        self.camera_list = camera_list if camera_list else ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]

    def before_step(self, action, physics):
        left_arm_action = action[:6]
        right_arm_action = action[8:8 + 6]
        normalized_left_gripper_action = action[6]
        normalized_right_gripper_action = action[8 + 6]

        # Clamp gripper values to 0.3 or 0.65
        # if normalized_left_gripper_action < 0.028:
        #     normalized_left_gripper_action = 0.02
        # else:
        #     normalized_left_gripper_action = 0.065

        # if normalized_right_gripper_action < 0.028:
        #     normalized_right_gripper_action = 0.02
        # else:
        #     normalized_right_gripper_action = 0.065

        # Assign the processed gripper actions
        left_gripper_action = normalized_left_gripper_action
        right_gripper_action = normalized_right_gripper_action

        # Ensure both gripper fingers act oppositely
        full_left_gripper_action = [left_gripper_action, left_gripper_action]
        full_right_gripper_action = [right_gripper_action, right_gripper_action]

        # Concatenate the final action array
        env_action = np.concatenate([
            left_arm_action,
            full_left_gripper_action,
            right_arm_action,
            full_right_gripper_action,
        ])
        # print(f"action: {action}")
        # print(f"env_action: {env_action}")
        super().before_step(env_action, physics)

        return
    
    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)


    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()
        return env_state

    def get_position(self, physics):
        position = physics.data.qpos.copy()
        return position[:16]

    def get_velocity(self, physics):
        velocity = physics.data.qvel.copy()
        return velocity[:16]

    def get_observation(self, physics) -> collections.OrderedDict:
        obs = get_observation_base(physics, self.camera_list)
        obs['qpos'] = self.get_position(physics)
        obs['qvel'] = self.get_velocity(physics)
        obs['env_state'] = self.get_env_state(physics)
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        raise NotImplementedError

class TransferCubeTask(BimanualViperXTask):
    def __init__(self, random=None, onscreen_render=False, camera_list=None):
        super().__init__(random=random, onscreen_render=onscreen_render, camera_list=camera_list)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        # TODO Notice: this function does not randomize the env configuration. Instead, set
        # BOX_POSE from outside reset qpos, control and box position
        with physics.reset_context():
            physics.named.data.qpos[:16] = START_ARM_POSE
            # np.copyto(physics.data.ctrl, START_ARM_POSE)
            assert BOX_POSE[0] is not None
            physics.named.data.qpos[-7:] = BOX_POSE[0]
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[16:]
        return env_state

    def get_reward(self, physics):
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

def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    # setup the environment
    camera_list = ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    env = make_sim_env(TransferCubeTask, 'aloha_scene_joint.xml')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    plt_imgs = plot_observation_images(ts.observation, camera_list)

    for t in range(1000):
        action = np.random.uniform(-np.pi, np.pi, 16)
        ts = env.step(action)
        episode.append(ts)

        plt_imgs[0].set_data(ts.observation['images']['camera_high'])
        plt_imgs[1].set_data(ts.observation['images']['camera_low'])
        plt_imgs[2].set_data(ts.observation['images']['camera_left_wrist'])
        plt_imgs[3].set_data(ts.observation['images']['camera_right_wrist'])

        plt.pause(0.02)

if __name__ == '__main__':
    BOX_POSE = sample_box_pose()

    test_sim_teleop()
