import numpy as np
import os
import collections
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
import importlib
import matplotlib.pyplot as plt
from constants import XML_DIR, DT

def sample_box_pose():
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.02, 0.02]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def get_observation_base(physics, camera_list, on_screen_render=True):
    obs = collections.OrderedDict()
    if on_screen_render:
        obs["images"] = dict()
        for cam in camera_list:
            obs['images'][cam] = physics.render(height=480, width=640, camera_id=cam)
    return obs

def make_sim_env(task_class, xml_file='trossen_ai_scene.xml', task_name='sim_transfer_cube', onscreen_render=False, camera_list=None):
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
        xml_path = os.path.join(XML_DIR, xml_file)
        physics = mujoco.Physics.from_xml_path(xml_path)
        task = task_class(random=False, onscreen_render=onscreen_render, camera_list = camera_list)
    else:
        raise NotImplementedError

    env = control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )
    return env

def plot_observation_images(observation, camera_list):
    images = observation.get("images", {})
    
    # Define the layout dynamically based on the provided camera list
    num_cameras = len(camera_list)

    if num_cameras == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, num_cameras)  # Maximum of 3 columns
        rows = (num_cameras + cols - 1) // cols  # Compute rows dynamically
    fig, axs = plt.subplots(rows, cols, figsize=(10, 10))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]
    
    plt_imgs = []
    titles = {
        "camera_high": "Camera High",
        "camera_low": "Camera Low",
        "camera_teleop": "Teleoperator POV",
        "camera_left_wrist": "Left Wrist Camera",
        "camera_right_wrist": "Right Wrist Camera",
    }
    
    for i, cam in enumerate(camera_list):
        if cam in images:
            plt_imgs.append(axs[i].imshow(images[cam]))
            axs[i].set_title(titles.get(cam, cam))
    
    for ax in axs.flat:
        ax.axis("off")
    
    plt.ion()
    return plt_imgs

def set_observation_images(observation, plt_imgs, camera_list):
    images = observation.get("images", {})

    # Update image data dynamically
    for i, cam in enumerate(camera_list):
        if cam in images and i < len(plt_imgs):  
            plt_imgs[i].set_data(images[cam])

    plt.pause(0.02)
    return plt_imgs