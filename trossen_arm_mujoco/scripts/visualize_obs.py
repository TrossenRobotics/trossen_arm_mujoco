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

import argparse
import os
import re

import cv2
import h5py
import numpy as np
from trossen_arm_mujoco.constants import ROOT_DIR, DATA_DIR, USER_NAME


def load_hdf5(dataset_path: str) -> dict | None:
    """
    Load camera feed data from an HDF5 dataset.

    :param dataset_path: Path to the HDF5 dataset file.
    :return: Dictionary mapping camera names to image sequences.
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at {dataset_path}")
        return None

    with h5py.File(dataset_path, "r") as root:
        image_dict = {}
        for cam_name in root["/observations/images/"].keys():
            image_dict[cam_name] = root[f"/observations/images/{cam_name}"][()]

    return image_dict


def save_videos(image_dict: dict, dt: float, video_path: str) -> None:
    """
    Save all camera feeds into a single MP4 video.

    :param image_dict: Dictionary mapping camera names to image sequences.
    :param dt: Time interval per frame (1 / fps).
    :param video_path: Path to save the output MP4 video.
    """
    if not image_dict:
        print(f"Skipping {video_path}: No valid images found.")
        return

    cam_names = list(image_dict.keys())
    h, w, _ = image_dict[cam_names[0]][0].shape
    # Concatenate all cameras horizontally
    w_total = w * len(cam_names)
    fps = int(1 / dt)

    # Initialize video writer
    out = cv2.VideoWriter(
        video_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w_total, h),
    )

    num_frames = len(image_dict[cam_names[0]])

    for frame_idx in range(num_frames):
        # Convert RGB to BGR
        frame_row = [
            image_dict[cam_name][frame_idx][:, :, [2, 1, 0]] for cam_name in cam_names
        ]
        # Horizontally concatenate
        concatenated_frame = np.concatenate(frame_row, axis=1)
        out.write(concatenated_frame)

    out.release()
    print(f"Saved video to: {video_path}")


def process_directory(args: argparse.Namespace):
    """
    Convert all HDF5 episodes in the dataset directory to MP4 videos.

    :param hdf5_dir: Path to the directory containing HDF5 dataset files.
    :param output_dir: Path to the directory to save the output MP4 videos.
    :param fps: Frames per second for the output video, defaults to 50.
    """
    root_dir = args.root_dir if args.root_dir else ROOT_DIR
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    output_dir = args.output_dir
    fps = args.fps

    hdf5_dir = os.path.join(root_dir, USER_NAME, data_dir)
    output_dir = os.path.join(root_dir, USER_NAME, output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # Match episode files (episode_{number}.hdf5)
    episode_pattern = re.compile(r"episode_(\d+)\.hdf5")

    # Iterate over all files in the directory
    for filename in os.listdir(hdf5_dir):
        match = episode_pattern.match(filename)
        if match:
            episode_number = match.group(1)
            input_path = os.path.join(hdf5_dir, filename)
            print(f"Processing {input_path}")
            output_path = os.path.join(output_dir, f"episode_{episode_number}.mp4")

            print(f"Processing {input_path} → {output_path}")

            # Load camera data
            images = load_hdf5(input_path)
            if images:
                # Save to MP4 video
                save_videos(images, dt=1 / fps, video_path=output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert all HDF5 episodes in a directory to MP4 videos.",
    )
    parser.add_argument(
        "--data_dir",
        help="Path to the directory containing HDF5 dataset files.",
    )
    parser.add_argument(
        "--root_dir",
        help="Root directory for data saving.",
    )
    parser.add_argument(
        "--output_dir",
        default="videos",
        help="Path to the directory to save the output MP4 videos.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Frames per second for the video. Default is 50.",
    )

    args = parser.parse_args()

    # Process all episodes in the directory
    process_directory(args)
