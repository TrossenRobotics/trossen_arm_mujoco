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
"""Stationary AI Transfer Cube Policy Evaluation.

Runs a policy on randomized cube positions and reports the success rate of each subtask.

A policy is any object with reset() and __call__(observation) -> 14 joint targets.
To evaluate your own policy (e.g. a trained ACT model), add it to make_policy().

Usage:
    uv run stationary_ai_transfer_cube_eval --policy scripted --num_trials 50

Options:
    --policy NAME          Policy to evaluate (default: scripted)
    --num_trials N         Number of evaluation episodes (default: 50)
    --episode_len N        Steps per episode at 50 Hz (default: 1200, i.e. 24 s)
    --seed N               Random seed for cube positions (default: 1000, different from recording)
    --onscreen_render      Show the simulation in the MuJoCo viewer
    --cam_names [CAM ...]  Cameras the policy uses, separated by spaces (default: all four)
                           Available: cam_high cam_low follower_left_camera follower_right_camera
                           e.g. --cam_names cam_high           only the top camera
                                --cam_names                    no cameras (faster, for policies that
                                                               do not use images, like scripted)
"""

from __future__ import annotations

import argparse
import sys

import mujoco
import mujoco.viewer
import numpy as np
from trossen_arm_mujoco.scripts.stationary_ai_transfer_cube_record import DEFAULT_EPISODE_LEN, ScriptedPolicy

from trossen_arm_mujoco.transfer_cube_env import MAX_REWARD, TransferCubeEnv

SUBTASK_NAMES = ["Touch", "Lift", "Handover", "Place"]  # reached at reward 1, 2, 3, 4


def make_policy(name: str, env: TransferCubeEnv):
    """Create the policy to evaluate."""
    if name == "scripted":
        return ScriptedPolicy(env)
    raise ValueError(f"Unknown policy: {name}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a policy on the transfer cube task.")
    parser.add_argument("--policy", type=str, default="scripted", help="Policy to evaluate.")
    parser.add_argument("--num_trials", type=int, default=50, help="Number of evaluation episodes.")
    parser.add_argument("--episode_len", type=int, default=DEFAULT_EPISODE_LEN, help="Steps per episode.")
    parser.add_argument("--seed", type=int, default=1000, help="Random seed for cube positions.")
    parser.add_argument("--onscreen_render", action="store_true", help="Show the simulation in the viewer.")
    parser.add_argument("--cam_names", type=str, nargs="*", default=None, help="Cameras the policy uses.")
    args = parser.parse_args()

    print(f"Evaluating '{args.policy}' policy on {args.num_trials} transfer cube episodes")

    env = TransferCubeEnv(camera_names=args.cam_names)
    env.reset(seed=args.seed)
    policy = make_policy(args.policy, env)

    viewer = mujoco.viewer.launch_passive(env.model, env.data) if args.onscreen_render else None

    # Highest reward reached in each episode
    max_rewards = []
    for trial in range(args.num_trials):
        observation = env.reset()
        policy.reset()

        max_reward = 0
        for _ in range(args.episode_len):
            observation, reward = env.step(policy(observation))
            max_reward = max(max_reward, reward)
            if viewer is not None:
                viewer.sync()

        # Success requires the final state to be placed, not just reaching it
        max_rewards.append(max_reward if reward == MAX_REWARD else min(max_reward, MAX_REWARD - 1))
        print(f"Trial {trial}: reward {max_rewards[-1]}/{MAX_REWARD}")

    if viewer is not None:
        viewer.close()
    env.close()

    max_rewards = np.array(max_rewards)
    print("\nSubtask success rates:")
    for stage, name in enumerate(SUBTASK_NAMES, start=1):
        print(f"  {name:<9} {100 * np.mean(max_rewards >= stage):5.1f}%")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nStopping evaluation...")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
