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
"""Stationary AI Transfer Cube Demonstration Recording.

Records scripted policy demonstrations of the transfer cube task as HDF5 episodes.

Usage:
    uv run stationary_ai_transfer_cube_record --num_episodes 50

Options:
    --root_dir DIR       Root folder for datasets (default: data/ in the current directory)
    --data_dir NAME      Dataset folder inside root_dir (default: stationary_ai_transfer_cube)
    --num_episodes N     Number of successful episodes to save (default: 50)
    --episode_len N      Steps per episode at 50 Hz (default: 1200, i.e. 24 s)
    --seed N             Random seed for cube positions and noise (default: 0)
    --inject_noise       Add a smooth random offset of up to ±1 cm to the scripted path
    --onscreen_render    Show the simulation in the MuJoCo viewer
    --cam_names CAM ...  Cameras to record, separated by spaces (default: all four)
                         Available: cam_high cam_low follower_left_camera follower_right_camera
                         e.g. --cam_names cam_high cam_low

Note: Each episode should take about 30 s to record. If it is much slower, MuJoCo is rendering
on the CPU; see "Slow Recording" in the README.
"""

from __future__ import annotations

import argparse
import os
import sys

import h5py
import mujoco
import mujoco.viewer
import numpy as np
from trossen_arm_mujoco.scripts.stationary_ai_pick_place import DEFAULT_EVENTS_DT, StationaryAIPickPlace

from trossen_arm_mujoco.transfer_cube_env import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    MAX_REWARD,
    TARGET_POSITION,
    TransferCubeEnv,
)

# Recording configuration
ROOT_DIR = "data"
DEFAULT_DATA_DIR = "stationary_ai_transfer_cube"
DEFAULT_EPISODE_LEN = 1200  # 24 s at 50 Hz
# Demo phases run TIME_SCALE times slower than in the interactive demo, for realistic arm speeds
TIME_SCALE = 2
# With --inject_noise, the end effector path gets a smooth random offset of up to ±NOISE_SCALE
# in x, y and z (a new offset at each phase keyframe, interpolated in between). The demonstrations
# then vary slightly instead of being identical, so a policy trained on them also sees small
# deviations from the ideal path.
NOISE_SCALE = 0.01  # meters


class ScriptedPolicy:
    """Stationary AI pick-and-place demo running as a 50 Hz policy."""

    def __init__(self, env: TransferCubeEnv, inject_noise: bool = False):
        """Initialize scripted policy.

        :param env: Environment to control.
        :param inject_noise: Add a smooth random offset to the end effector path.
        """
        self.env = env
        self.inject_noise = inject_noise

        # The demo advances one waypoint per call, so scale its phase timing to 50 Hz
        self.task = StationaryAIPickPlace(
            events_dt=[n * TIME_SCALE // env.n_substeps for n in DEFAULT_EVENTS_DT],
            target_position=TARGET_POSITION,
        )
        self.task.setup_scene(env.model, env.data)

    def reset(self) -> None:
        """Plan the trajectory for the current cube position."""
        self.task.reset_robot()
        self.task.generate_dual_arm_trajectory()

        if self.inject_noise:
            # Random offset at each phase keyframe (none at the start), interpolated in between
            keyframe_steps = np.concatenate([[0], np.cumsum(self.task.events_dt)])
            for name in ["left_trajectory", "right_trajectory"]:
                offsets = self.env.rng.uniform(-NOISE_SCALE, NOISE_SCALE, (len(keyframe_steps), 3))
                offsets[0] = 0.0
                trajectory = getattr(self.task, name)
                noisy = [
                    (pos + [np.interp(step, keyframe_steps, offsets[:, k]) for k in range(3)], ori, step)
                    for pos, ori, step in trajectory
                ]
                setattr(self.task, name, noisy)

    def __call__(self, observation: dict) -> np.ndarray:
        """Compute the next 14 joint targets with IK."""
        self.task.forward()
        return self.env.data.ctrl.copy()


def record_episode(
    env: TransferCubeEnv,
    policy: ScriptedPolicy,
    episode_len: int,
    path: str,
    viewer: mujoco.viewer.Handle | None = None,
) -> bool:
    """Run one episode and write it to an HDF5 file.

    :return: True if the episode succeeded (the file is deleted otherwise).
    """
    observation = env.reset()
    policy.reset()

    with h5py.File(path, "w") as root:
        root.attrs["sim"] = True
        qpos = root.create_dataset("observations/qpos", (episode_len, 14), dtype="float32")
        qvel = root.create_dataset("observations/qvel", (episode_len, 14), dtype="float32")
        action = root.create_dataset("action", (episode_len, 14), dtype="float32")
        images = {
            name: root.create_dataset(
                f"observations/images/{name}",
                (episode_len, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                dtype="uint8",
                chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                compression="gzip",
                compression_opts=1,
            )
            for name in env.camera_names
        }

        for t in range(episode_len):
            # Record observation and the action taken from it
            qpos[t] = observation["qpos"]
            qvel[t] = observation["qvel"]
            for name, image in observation["images"].items():
                images[name][t] = image
            action[t] = policy(observation)

            observation, reward = env.step(action[t])
            if viewer is not None:
                viewer.sync()

    success = reward == MAX_REWARD
    if not success:
        os.remove(path)
    return success


def main():
    parser = argparse.ArgumentParser(description="Record scripted transfer cube demonstrations.")
    parser.add_argument("--root_dir", type=str, default=ROOT_DIR, help="Root directory for saving data.")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="Dataset directory name.")
    parser.add_argument("--num_episodes", type=int, default=50, help="Number of successful episodes to save.")
    parser.add_argument("--episode_len", type=int, default=DEFAULT_EPISODE_LEN, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for cube positions and noise.")
    parser.add_argument("--inject_noise", action="store_true", help="Add a smooth random offset to the scripted path.")
    parser.add_argument("--onscreen_render", action="store_true", help="Show the simulation in the viewer.")
    parser.add_argument("--cam_names", type=str, nargs="+", default=None, help="Cameras to record.")
    args = parser.parse_args()

    data_dir = os.path.join(args.root_dir, args.data_dir)
    os.makedirs(data_dir, exist_ok=True)
    print(f"Recording {args.num_episodes} transfer cube episodes to {data_dir}")

    env = TransferCubeEnv(camera_names=args.cam_names)
    env.reset(seed=args.seed)
    policy = ScriptedPolicy(env, inject_noise=args.inject_noise)

    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.onscreen_render else None

    # Each run records one full episode with a new random cube position. Failed runs are discarded,
    # so keep running until num_episodes are saved (give up after 2 x num_episodes runs).
    saved = 0
    runs = 0
    while saved < args.num_episodes and runs < 2 * args.num_episodes:
        path = os.path.join(data_dir, f"episode_{saved}.hdf5")
        success = record_episode(env, policy, args.episode_len, path, viewer)
        runs += 1
        if success:
            saved += 1
            print(f"Run {runs}: success, saved {path}")
        else:
            print(f"Run {runs}: failed (cube not placed on target), discarded")

    if viewer is not None:
        viewer.close()
    env.close()

    print(f"\nSaved {saved}/{args.num_episodes} episodes in {runs} runs ({runs - saved} failed)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping recording...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
