import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np
from utils import get_observation_base, make_sim_env, plot_observation_images, set_observation_images

XML_DIR = "assets"
DT = 0.02
BOX_POSE = [None]  # to be changed from outside
RADIUS = 0.05  # Radius of the circular trajectory
CENTER_LEFT = np.array([-0.3, 0.0, 0.4])  # Center for the left end-effector
CENTER_RIGHT = np.array([0.3, 0.0, 0.4])  # Center for the right end-effector

class BimanualViperXTask(base.Task):
    def __init__(self, random=None, onscreen_render=False):
        super().__init__(random=random)
        self.on_screen_render = onscreen_render

    def initialize_episode(self, physics):
        """Sets the state of the environment at the start of each episode."""
        super().initialize_episode(physics)

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()
        return env_state

    def get_observation(self, physics) -> collections.OrderedDict:
        obs = get_observation_base(physics)
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        obs["env_state"] = self.get_env_state(physics)
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        return 0.0


def circular_motion(t, center, radius, frequency=0.5):
    """
    Generates a circular motion trajectory for the end-effector.
    """
    x = center[0] + radius * np.cos(2 * np.pi * frequency * t)
    y = center[1] + radius * np.sin(2 * np.pi * frequency * t)
    z = center[2]  # Keep the z-coordinate constant
    return np.array([x, y, z])

def get_observation(physics) -> collections.OrderedDict:
        obs = get_observation_base(physics)
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        return obs

def test_sim_mocap_control():
    """Testing teleoperation in sim with ALOHA using mocap."""
    # Setup the environment
    env = make_sim_env(BimanualViperXTask, "aloha_scene.xml")
    ts = env.reset()
    physics = env.physics

    # Setup plotting
    plt_imgs = plot_observation_images(ts.observation, 5)

    # Time variable for circular motion
    t = 0

    for step in range(1000):
        # Compute circular motion for left and right end-effectors
        mocap_left_pos = circular_motion(t, CENTER_LEFT, RADIUS)
        mocap_right_pos = circular_motion(t, CENTER_RIGHT, RADIUS, frequency=0.7)  # Slightly different frequency

        # Update mocap positions in the simulation
        np.copyto(physics.data.mocap_pos[0], mocap_left_pos)
        np.copyto(physics.data.mocap_pos[1], mocap_right_pos)

        # Keep the orientation of the end-effectors fixed
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])  # Identity quaternion
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])  # Identity quaternion

        physics.step()
        # Step the environment
        obs = get_observation(physics)

        # Update images
        plt_imgs = set_observation_images(obs, plt_imgs)

        # Increment time for circular motion
        t += DT

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    test_sim_mocap_control()
