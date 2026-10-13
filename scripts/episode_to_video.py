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
"""HDF5 Episode to Video.

Converts one recorded episode to an MP4 with all cameras side by side.

Usage:
    uv run python3 scripts/episode_to_video.py data/stationary_ai_transfer_cube/episode_0.hdf5

Options:
    --fps N  Video frame rate (default: 50, the recording rate)
"""

import argparse
import os

import cv2
import h5py
import numpy as np


def episode_to_video(episode_path: str, fps: int) -> str:
    """Save all camera feeds of an episode into a single MP4 video.

    :param episode_path: Path to the episode HDF5 file.
    :param fps: Frames per second of the video.
    :return: Path of the saved video.
    """
    video_path = os.path.splitext(episode_path)[0] + ".mp4"

    with h5py.File(episode_path, "r") as root:
        cameras = [root[f"observations/images/{name}"] for name in root["observations/images"]]
        num_frames, height, width, _ = cameras[0].shape

        writer = cv2.VideoWriter(
            video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width * len(cameras), height)
        )
        for t in range(num_frames):
            # Cameras side by side, RGB -> BGR for OpenCV
            frame = np.concatenate([camera[t] for camera in cameras], axis=1)
            writer.write(frame[:, :, ::-1])
        writer.release()

    return video_path


def main():
    parser = argparse.ArgumentParser(description="Convert one HDF5 episode to an MP4 video.")
    parser.add_argument("episode_path", type=str, help="Path to the episode HDF5 file.")
    parser.add_argument("--fps", type=int, default=50, help="Video frame rate (default: 50).")
    args = parser.parse_args()

    print(f"Converting {args.episode_path}")
    video_path = episode_to_video(args.episode_path, args.fps)
    print(f"Saved video to {video_path}")


if __name__ == "__main__":
    main()
