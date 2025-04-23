import h5py
import numpy as np
import time
from pathlib import Path
import argparse
import trossen_arm
import os
from trossen_arm_mujoco.constants import ROOT_DIR, DATA_DIR, USER_NAME
from pathlib import Path


def configure_arm(ip_address, home_pose):
    """Configure a Trossen arm and set to home pose."""
    driver = trossen_arm.TrossenArmDriver()
    driver.configure(
        trossen_arm.Model.wxai_v0,
        trossen_arm.StandardEndEffector.wxai_v0_leader,
        ip_address,
        False
    )
    driver.set_all_modes(trossen_arm.Mode.position)
    driver.set_all_positions(home_pose, 2.0, True)
    return driver


def replay_episode(hdf5_path, left_driver, right_driver, home_pose, fps=10):
    """Replay actions from an HDF5 episode file."""
    with h5py.File(hdf5_path, "r") as f:
        actions = f["action"]
        for t in range(actions.shape[0]):
            action_t = actions[t]
            left_action, right_action = np.split(action_t, 2)
            left_action = left_action[:-1]
            right_action = right_action[:-1]

            print(f"Timestep {t}: Left = {left_action}, Right = {right_action}")
            right_driver.set_all_positions(right_action, 3.0 / fps, False)
            left_driver.set_all_positions(left_action, 3.0 / fps, False)
            time.sleep(1 / fps)


def cleanup_arms(left_driver, right_driver, home_pose):
    """Send both arms to home and then sleep position."""
    left_driver.set_all_positions(home_pose, 2.0, False)
    right_driver.set_all_positions(home_pose, 2.0, False)

    sleep_pose = np.zeros(left_driver.get_num_joints() - 1)
    left_driver.set_arm_positions(sleep_pose, 2.0, True)
    right_driver.set_arm_positions(sleep_pose, 2.0, True)


def main(args):
    home_pose = np.array([0.0, np.pi / 12, np.pi / 12, 0.0, 0.0, 0.0, 0.0])
    root_dir = args.root_dir if args.root_dir else ROOT_DIR
    data_dir = args.data_dir if args.data_dir else DATA_DIR
    hdf5_path = os.path.join(root_dir, USER_NAME, data_dir, f"episode_{args.episode_idx}.hdf5")

    if not Path(hdf5_path).exists():
        print(f"HDF5 file not found at {hdf5_path}")
        return

    print(f"Configuring left arm at {args.left_ip} and right arm at {args.right_ip}...")
    left_driver = configure_arm(args.left_ip, home_pose)
    right_driver = configure_arm(args.right_ip, home_pose)

    print(f"Replaying episode from {hdf5_path}...")
    replay_episode(hdf5_path, left_driver, right_driver, home_pose, args.fps)

    print("Cleaning up arms...")
    cleanup_arms(left_driver, right_driver, home_pose)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay HDF5 episode for Trossen arms.")
    parser.add_argument("--data_dir", type=str, help="Dataset directory name.")
    parser.add_argument("--root_dir", type=str, help="Root directory for data.")
    parser.add_argument("--episode_idx", type=int, default=0, help="Episode index to replay.")
    parser.add_argument("--fps", type=int, default=10, help="Playback frames per second.")
    parser.add_argument("--left_ip", type=str, default="192.168.1.5", help="IP address of left arm.")
    parser.add_argument("--right_ip", type=str, default="192.168.1.4", help="IP address of right arm.")

    args = parser.parse_args()
    main(args)