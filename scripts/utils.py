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
    """
    Generate a random pose for a cube within predefined position ranges.

    :return: A 7D array containing the sampled position ``[x, y, z, w, x, y, z]`` 
        representing the cube's position and orientation as a quaternion.
    :rtype: np.ndarray
    """
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.02, 0.02]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def get_observation_base(physics, camera_list, on_screen_render=True):
    """
    Capture image observations from multiple cameras in the simulation.

    :param physics: The simulation physics instance.
    :type physics: mujoco.Physics
    :param camera_list: List of camera names to capture images from.
    :type camera_list: list
    :param on_screen_render: Whether to capture images from cameras, defaults to ``True``.
    :type on_screen_render: bool, optional
    :return: A dictionary containing image observations.
    :rtype: collections.OrderedDict
    """
    obs = collections.OrderedDict()
    if on_screen_render:
        obs["images"] = dict()
        for cam in camera_list:
            obs['images'][cam] = physics.render(height=480, width=640, camera_id=cam)
    return obs

def make_sim_env(task_class, xml_file='trossen_ai_scene.xml', task_name='sim_transfer_cube', onscreen_render=False, camera_list=None):
    """
    Create a simulated environment for bi-manual robotic manipulation.

    :param task_class: The task class for defining simulation behavior.
    :type task_class: class
    :param xml_file: Path to the robot XML file, defaults to ``'trossen_ai_scene.xml'``.
    :type xml_file: str, optional
    :param task_name: Name of the task, defaults to ``'sim_transfer_cube'``.
    :type task_name: str, optional
    :param onscreen_render: Whether to render the simulation on-screen, defaults to ``False``.
    :type onscreen_render: bool, optional
    :param camera_list: List of camera names to be used, defaults to ``None``.
    :type camera_list: list, optional
    :return: The simulated robot environment.
    :rtype: control.Environment
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
    """
    Plot observation images from multiple camera viewpoints.

    :param observation: The observation data containing images.
    :type observation: dict
    :param camera_list: List of camera names used for capturing images.
    :type camera_list: list
    :return: A list of :class:`matplotlib.image.AxesImage` objects for dynamic updates.
    :rtype: list
    """
    images = observation.get("images", {})
    
    # Define the layout based on the provided camera list
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
    """
    Update displayed observation images dynamically.

    :param observation: The observation data containing updated images.
    :type observation: dict
    :param plt_imgs: A list of :class:`matplotlib.image.AxesImage` objects for dynamic updates.
    :type plt_imgs: list
    :param camera_list: List of camera names.
    :type camera_list: list
    :return: Updated list of :class:`matplotlib.image.AxesImage` objects for real-time visualization.
    :rtype: list
    """
    images = observation.get("images", {})

    # Update image data dynamically
    for i, cam in enumerate(camera_list):
        if cam in images and i < len(plt_imgs):  
            plt_imgs[i].set_data(images[cam])

    plt.pause(0.02)
    return plt_imgs
