import collections
import os


from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import numpy as np
import matplotlib.pyplot as plt


from utils import sample_box_pose

XML_DIR="assets"
DT = 0.02
BOX_POSE = [None] # to be changed from outside
START_ARM_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # Change to 16 for bimanual

def make_ee_sim_env(task_name: str):
    """
    Environment for simulated robot bi-manual manipulation, with end-effector control.

    Action space: [
        left_arm_pose (7),              # position and quaternion for end effector
        left_gripper_positions (1),     # normalized gripper position (0: close, 1: open)
        right_arm_pose (7),             # position and quaternion for end effector
        right_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
    ]

    Observation space: {
        "qpos": Concat[
            left_arm_qpos (6),          # absolute joint position (rad)
            left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
            right_arm_qpos (6),         # absolute joint position (rad)
            right_gripper_qpos (1)]     # normalized gripper position (0: close, 1: open)
        ],
        "qvel": Concat[
            left_arm_qvel (6),          # absolute joint velocity (rad/s)
            left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
            right_arm_qvel (6),         # absolute joint velocity (rad/s)
            right_gripper_qvel (1)]     # normalized gripper velocity (pos: opening, neg: closing)
        ],
        "images": {"main": (480x640x3)} # h, w, c, dtype='uint8'
    }
    """
    if 'sim_transfer_cube' in task_name:
        xml_path = os.path.join(XML_DIR, 'aloha_scene.xml')
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = TransferCubeEETask(random=False)
   
    else:
        raise NotImplementedError
    return control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )


class BimanualViperXEETask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)

    def before_step(self, action, physics):
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
        # reset joint position
        physics.named.data.qpos[:12] = START_ARM_POSE

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
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)


    @staticmethod
    def get_env_state(physics):
        raise NotImplementedError

    def get_position(self, physics):
        positions = physics.data.qpos.copy()
        return positions[:16]
    
    def get_velocity(self, physics):
        velocities = physics.data.qvel.copy()
        return velocities[:16]
    
    def get_observation(self, physics):
        # note: it is important to do .copy()
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_position(physics)
        obs['qvel'] = self.get_velocity(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['camera_high'] = physics.render(height=480, width=640, camera_id='camera_high')
        obs['images']['camera_low'] = physics.render(height=480, width=640, camera_id='camera_low')
        obs['images']['camera_left_wrist'] = physics.render(height=480, width=640, camera_id='camera_left_wrist')
        obs['images']['camera_right_wrist'] = physics.render(height=480, width=640, camera_id='camera_right_wrist')
        # used in scripted policy to obtain starting pose
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
        raise NotImplementedError


class TransferCubeEETask(BimanualViperXEETask):
    def __init__(self, random=None):
        super().__init__(random=random)
        self.max_reward = 4

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        self.initialize_robots(physics)
        # randomize box position
        cube_pose = sample_box_pose()
        box_start_idx = physics.model.name2id('red_box_joint', 'joint')
        np.copyto(physics.data.qpos[box_start_idx : box_start_idx + 7], cube_pose)

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


def test_ee_sim_env():
     # setup the environment
    env = make_ee_sim_env('sim_transfer_cube')
    # print(f"Action space: {env.action_spec().shape}")
    ts = env.reset()
    episode = [ts]
    # setup plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt_imgs = [
        axs[0, 0].imshow(ts.observation['images']['camera_high']),
        axs[0, 1].imshow(ts.observation['images']['camera_low']),
        axs[1, 0].imshow(ts.observation['images']['camera_left_wrist']),
        axs[1, 1].imshow(ts.observation['images']['camera_right_wrist']),
    ]

    # Optionally, add titles for better clarity
    axs[0, 0].set_title("Camera High")
    axs[0, 1].set_title("Camera Low")
    axs[1, 0].set_title("Left Wrist Camera")
    axs[1, 1].set_title("Right Wrist Camera")


    # Remove axis ticks for better visualization
    for ax in axs.flat:
        ax.axis('off')

    # plt.tight_layout()
    plt.ion()

    for t in range(1000):
        action = np.random.uniform(-np.pi, np.pi, 14)
        ts = env.step(action)
        episode.append(ts)

        plt_imgs[0].set_data(ts.observation['images']['camera_high'])
        plt_imgs[1].set_data(ts.observation['images']['camera_low'])
        plt_imgs[2].set_data(ts.observation['images']['camera_left_wrist'])
        plt_imgs[3].set_data(ts.observation['images']['camera_right_wrist'])



        plt.pause(0.02)



if __name__ == '__main__':
    test_ee_sim_env()
