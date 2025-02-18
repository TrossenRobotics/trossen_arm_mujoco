import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np

XML_DIR = "assets"
DT = 0.02
BOX_POSE = [None]  # to be changed from outside
RADIUS = 0.05  # Radius of the circular trajectory
CENTER_LEFT = np.array([-0.3, 0.0, 0.4])  # Center for the left end-effector
CENTER_RIGHT = np.array([0.3, 0.0, 0.4])  # Center for the right end-effector

def make_sim_env(task_name: str):
    """
    Environment for simulated robot bi-manual manipulation, with joint position control.
    """
    xml_path = os.path.join(XML_DIR, "aloha_scene.xml")
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
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        obs["env_state"] = self.get_env_state(physics)
        obs["images"] = dict()
        obs["images"]["camera_high"] = physics.render(height=480, width=640, camera_id="camera_high")
        obs["images"]["camera_low"] = physics.render(height=480, width=640, camera_id="camera_low")
        obs["images"]["camera_left_wrist"] = physics.render(height=480, width=640, camera_id="camera_left_wrist")
        obs["images"]["camera_right_wrist"] = physics.render(height=480, width=640, camera_id="camera_right_wrist")
        obs["images"]["camera_teleop"] = physics.render(height=480, width=640, camera_id="teleoperator_pov")

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
        obs = collections.OrderedDict()
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        obs["images"] = dict()
        obs["images"]["camera_high"] = physics.render(height=480, width=640, camera_id="camera_high")
        obs["images"]["camera_low"] = physics.render(height=480, width=640, camera_id="camera_low")
        obs["images"]["camera_left_wrist"] = physics.render(height=480, width=640, camera_id="camera_left_wrist")
        obs["images"]["camera_right_wrist"] = physics.render(height=480, width=640, camera_id="camera_right_wrist")
        obs["images"]["camera_teleop"] = physics.render(height=480, width=640, camera_id="teleoperator_pov")

        return obs

def test_sim_mocap_control():
    """Testing teleoperation in sim with ALOHA using mocap."""
    # Setup the environment
    env = make_sim_env("sim_transfer_cube")
    ts = env.reset()
    physics = env.physics

    # Setup plotting
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    plt_imgs = [
        axs[0, 0].imshow(ts.observation["images"]["camera_high"]),
        axs[0, 1].imshow(ts.observation["images"]["camera_low"]),
        axs[0, 2].imshow(ts.observation["images"]["camera_teleop"]),
        axs[1, 0].imshow(ts.observation["images"]["camera_left_wrist"]),
        axs[1, 1].imshow(ts.observation["images"]["camera_right_wrist"]),
    ]

    axs[0, 0].set_title("Camera High")
    axs[0, 1].set_title("Camera Low")
    axs[0, 2].set_title("Teleoperator POV")
    axs[1, 0].set_title("Left Wrist Camera")
    axs[1, 1].set_title("Right Wrist Camera")

    for ax in axs.flat:
        ax.axis("off")

    plt.ion()

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
        plt_imgs[0].set_data(obs["images"]["camera_high"])
        plt_imgs[1].set_data(obs["images"]["camera_low"])
        plt_imgs[2].set_data(obs["images"]["camera_teleop"])
        plt_imgs[3].set_data(obs["images"]["camera_left_wrist"])
        plt_imgs[4].set_data(obs["images"]["camera_right_wrist"])

        plt.pause(DT)

        # Increment time for circular motion
        t += DT

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    test_sim_mocap_control()
