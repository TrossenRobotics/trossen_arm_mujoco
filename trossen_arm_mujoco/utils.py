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

import collections
import os

from dm_control import mujoco
from dm_control.mujoco import Physics
from dm_control.rl import control
from dm_control.suite import base
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
import numpy as np

from trossen_arm_mujoco.constants import ASSETS_DIR, DT


def sample_box_pose() -> np.ndarray:
    """
    Generate a random pose for a cube within predefined position ranges.

    :return: A 7D array containing the sampled position ``[x, y, z, w, x, y, z]`` representing the
        cube's position and orientation as a quaternion.
    """
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.02, 0.02]

    ranges = np.vstack([x_range, y_range, z_range])
    # cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_position = np.array([0.13, -0.19, 0])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def get_observation_base(
    physics: Physics,
    cam_list: list[str],
    on_screen_render: bool = True,
) -> collections.OrderedDict:
    """
    Capture image observations from multiple cameras in the simulation.

    :param physics: The simulation physics instance.
    :param cam_list: List of camera names to capture images from.
    :param on_screen_render: Whether to capture images from cameras, defaults to ``True``.
    :return: A dictionary containing image observations.
    """
    obs: collections.OrderedDict = collections.OrderedDict()
    if on_screen_render:
        obs["images"] = dict()
        for cam in cam_list:
            obs["images"][cam] = physics.render(height=480, width=640, camera_id=cam)
    return obs


def make_sim_env(
    task_class: base.Task,
    xml_file: str = "trossen_ai_scene.xml",
    task_name: str = "sim_transfer_cube",
    onscreen_render: bool = False,
    cam_list: list[str] = [],
):
    """
    Create a simulated environment for bimanual robotic manipulation.

    :param task_class: The task class for defining simulation behavior.
    :param xml_file: Path to the robot XML file, defaults to ``'trossen_ai_scene.xml'``.
    :param task_name: Name of the task, defaults to ``'sim_transfer_cube'``.
    :param onscreen_render: Whether to render the simulation on-screen, defaults to ``False``.
    :param cam_list: List of camera names to be used, defaults to ``[]``.
    :return: The simulated robot environment.
    """
    if "sim_transfer_cube" in task_name:
        assets_path = os.path.join(ASSETS_DIR, xml_file)
        physics = mujoco.Physics.from_xml_path(assets_path)
        task = task_class(
            random=False,
            onscreen_render=onscreen_render,
            cam_list=cam_list,
        )
    else:
        raise NotImplementedError(f"Task {task_name} is not implemented.")

    return control.Environment(
        physics,
        task,
        time_limit=20,
        control_timestep=DT,
        n_sub_steps=None,
        flat_observation=False,
    )


def plot_observation_images(observation: dict, cam_list: list[str]) -> list[AxesImage]:
    """
    Plot observation images from multiple camera viewpoints.

    :param observation: The observation data containing images.
    :param cam_list: List of camera names used for capturing images.
    :return: A list of AxesImage objects for dynamic updates.
    """
    images = observation.get("images", {})

    # Define the layout based on the provided camera list
    num_cameras = len(cam_list)

    if num_cameras == 4:
        cols = 2
        rows = 2
    else:
        cols = min(3, num_cameras)  # Maximum of 3 columns
        rows = (num_cameras + cols - 1) // cols  # Compute rows dynamically
    _, axs = plt.subplots(rows, cols, figsize=(10, 10))
    axs = axs.flatten() if isinstance(axs, (list, np.ndarray)) else [axs]

    plt_imgs: list[AxesImage] = []
    titles = {
        "cam_high": "Camera High",
        "cam_low": "Camera Low",
        "cam_teleop": "Teleoperator POV",
        "cam_left_wrist": "Left Wrist Camera",
        "cam_right_wrist": "Right Wrist Camera",
    }

    for i, cam in enumerate(cam_list):
        if cam in images:
            plt_imgs.append(axs[i].imshow(images[cam]))
            axs[i].set_title(titles.get(cam, cam))

    for ax in axs.flat:
        ax.axis("off")

    plt.ion()
    return plt_imgs


def set_observation_images(
    observation: dict,
    plt_imgs: list[AxesImage],
    cam_list: list[str],
) -> list[AxesImage]:
    """
    Update displayed observation images dynamically.

    :param observation: The observation data containing updated images.
    :param plt_imgs: A list of AxesImage objects for dynamic updates.
    :param cam_list: List of camera names.
    :return: Updated list of AxesImage objects for real-time visualization.
    """
    images = observation.get("images", {})

    # Update image data dynamically
    for i, cam in enumerate(cam_list):
        if cam in images and i < len(plt_imgs):
            plt_imgs[i].set_data(images[cam])

    plt.pause(0.02)
    return plt_imgs
