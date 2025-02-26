import collections
import os
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
import matplotlib.pyplot as plt
import numpy as np
from utils import get_observation_base

XML_DIR = "assets"
DT = 0.02


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
        obs = get_observation_base(physics)
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        return obs

    def get_reward(self, physics):
        # return whether left gripper is holding the box
        return 0.0

def get_observation(physics) -> collections.OrderedDict:
        obs = get_observation_base(physics)
        obs["qpos"] = physics.data.qpos.copy()
        obs["qvel"] = physics.data.qvel.copy()
        return obs

def interpolate_waypoints(waypoints, t, total_time):
    """
    Interpolate between waypoints for smooth motion.
    :param waypoints: List of 3D points defining the trajectory.
    :param t: Current time step.
    :param total_time: Total time to traverse the waypoints.
    :return: Interpolated position at time t.
    """
    num_segments = len(waypoints) - 1
    segment_time = total_time / num_segments
    current_segment = int(t // segment_time)
    
    if current_segment >= num_segments:
        return waypoints[-1]  # Stay at the last waypoint.

    t_segment = (t % segment_time) / segment_time
    return (1 - t_segment) * waypoints[current_segment] + t_segment * waypoints[current_segment + 1]

def test_sim_mocap_control():
    """Testing teleoperation in sim with ALOHA using mocap."""
    # Setup the environment
    env = make_sim_env("sim_transfer_cube")
    ts = env.reset()
    physics = env.physics

    # Define waypoints for both mocap bodies
    waypoints_left = [
        np.array([-0.3, 0.0, 0.4]),
        np.array([-0.3, 0.1, 0.4]),
        np.array([-0.2, 0.1, 0.4]),
        np.array([-0.2, 0.0, 0.4])
    ]
    waypoints_right = [
        np.array([0.3, 0.0, 0.4]),
        np.array([0.3, -0.1, 0.4]),
        np.array([0.2, -0.1, 0.4]),
        np.array([0.2, 0.0, 0.4])
    ]

    total_time = 5  # Time to traverse all waypoints
    t = 0  # Start time

    # Setup plotting
    fig, axs = plt.subplots(2, 3, figsize=(10, 10))
    plt_imgs = [
        axs[0, 0].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
        axs[0, 1].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
        axs[0, 2].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
        axs[1, 0].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
        axs[1, 1].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
        axs[1, 2].imshow(np.zeros((480, 640, 3), dtype=np.uint8)),
    ]

    axs[0, 0].set_title("Camera High")
    axs[0, 1].set_title("Camera Low")
    axs[0, 2].set_title("Teleoperator POV")
    axs[1, 0].set_title("Left Wrist Camera")
    axs[1, 1].set_title("Right Wrist Camera")

    for ax in axs.flat:
        ax.axis("off")

    plt.ion()

    while t < total_time:
        # Compute interpolated positions
        mocap_left_pos = interpolate_waypoints(waypoints_left, t, total_time)
        mocap_right_pos = interpolate_waypoints(waypoints_right, t, total_time)

        # Update mocap positions in the simulation
        np.copyto(physics.data.mocap_pos[0], mocap_left_pos)
        np.copyto(physics.data.mocap_pos[1], mocap_right_pos)

        # Keep the orientation of the end-effectors fixed
        np.copyto(physics.data.mocap_quat[0], [1, 0, 0, 0])  # Identity quaternion
        np.copyto(physics.data.mocap_quat[1], [1, 0, 0, 0])  # Identity quaternion

        physics.step()

        # Get updated observation
        obs = get_observation(physics)
        # print(obs["images"].keys())  # Debug print

        # Update images
        plt_imgs[0].set_data(obs["images"]["camera_high"])
        plt_imgs[1].set_data(obs["images"]["camera_low"])
        plt_imgs[2].set_data(obs["images"]["camera_teleop"])
        plt_imgs[3].set_data(obs["images"]["camera_left_wrist"])
        plt_imgs[4].set_data(obs["images"]["camera_right_wrist"])

        plt.pause(DT)

        # Increment time
        t += DT

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    test_sim_mocap_control()
