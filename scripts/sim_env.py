import collections
import os

from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np

XML_DIR="assets"
DT = 0.02
BOX_POSE = [None] # to be changed from outside
START_ARM_POSE = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

def make_sim_env(task_name: str):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control

    Action space: [
        left_arm_qpos (6),              # absolute joint position
        left_gripper_positions (1),     # normalized gripper position (0: close, 1: open)
        right_arm_qpos (6),             # absolute joint position
        right_gripper_positions (1),    # normalized gripper position (0: close, 1: open)
    ]

    Observation space: {
        "qpos": Concat[
            left_arm_qpos (6),          # absolute joint position
            left_gripper_position (1),  # normalized gripper position (0: close, 1: open)
            right_arm_qpos (6),         # absolute joint position
            right_gripper_qpos (1)      # normalized gripper position (0: close, 1: open)
        ],
        "qvel": Concat[
            left_arm_qvel (6),          # absolute joint velocity (rad)
            left_gripper_velocity (1),  # normalized gripper velocity (pos: opening, neg: closing)
            right_arm_qvel (6),         # absolute joint velocity (rad)
            right_gripper_qvel (1)      # normalized gripper velocity (pos: opening, neg: closing)
        ],
        "images": {"main": (480x640x3)  # h, w, c, dtype='uint8'
    }
    """
    
    xml_path = os.path.join(XML_DIR, 'aloha_scene.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = BimanualViperXTask(random=False)

    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env


class BimanualViperXTask(base.Task):
    def __init__(self, random=None):
        super().__init__(random=random)



    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

  
    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()
        return env_state

    def get_observation(self, physics) -> collections.OrderedDict:
        obs = collections.OrderedDict()
        obs['qpos'] = physics.data.qpos.copy()
        obs['qvel'] = physics.data.qvel.copy()
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        obs['images']['camera_high'] = physics.render(height=480, width=640, camera_id='camera_high')
        obs['images']['camera_low'] = physics.render(height=480, width=640, camera_id='camera_low')
        obs['images']['camera_left_wrist'] = physics.render(height=480, width=640, camera_id='camera_left_wrist')
        obs['images']['camera_right_wrist'] = physics.render(height=480, width=640, camera_id='camera_right_wrist')
        obs['images']['camera_teleop'] = physics.render(height=480, width=640, camera_id='teleoperator_pov')
        obs['images']['camera_collaborate'] = physics.render(height=480, width=640, camera_id='collaborator_pov')


        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        return 0.0



def test_sim_teleop():
    """ Testing teleoperation in sim with ALOHA. Requires hardware and ALOHA repo to work. """
    

    # setup the environment
    env = make_sim_env('sim_transfer_cube')
    ts = env.reset()
    episode = [ts]
    # setup plotting
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    plt_imgs = [
        axs[0, 0].imshow(ts.observation['images']['camera_high']),
        axs[0, 1].imshow(ts.observation['images']['camera_low']),
        axs[0, 2].imshow(ts.observation['images']['camera_teleop']),
        axs[1, 0].imshow(ts.observation['images']['camera_left_wrist']),
        axs[1, 1].imshow(ts.observation['images']['camera_right_wrist']),
        axs[1, 2].imshow(ts.observation['images']['camera_collaborate']),
    ]

    # Optionally, add titles for better clarity
    axs[0, 0].set_title("Camera High")
    axs[0, 1].set_title("Camera Low")
    axs[0, 2].set_title("Teleoperator POV")
    axs[1, 0].set_title("Left Wrist Camera")
    axs[1, 1].set_title("Right Wrist Camera")
    axs[1, 2].set_title("Collaborator POV")


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
        plt_imgs[3].set_data(ts.observation['images']['camera_left_wrist'])
        plt_imgs[4].set_data(ts.observation['images']['camera_right_wrist'])
        plt_imgs[2].set_data(ts.observation['images']['camera_teleop'])
        plt_imgs[5].set_data(ts.observation['images']['camera_collaborate'])


        plt.pause(0.02)



if __name__ == '__main__':
    test_sim_teleop()
