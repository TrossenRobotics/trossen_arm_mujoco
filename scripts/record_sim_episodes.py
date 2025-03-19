import time
import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import h5py

from ee_sim_env import TransferCubeEETask
from sim_env import BOX_POSE, TransferCubeTask
from scripted_policy import PickAndTransferPolicy
from tqdm import tqdm
from utils import make_sim_env
from utils import plot_observation_images, set_observation_images
from constants import SIM_TASK_CONFIGS

def main(args):
    """
    Generate demonstration data in simulation.
    First rollout the policy (defined in ee space) in ee_sim_env. Obtain the joint trajectory.
    Replace the gripper joint positions with the commanded joint position.
    Replay this joint trajectory (as action sequence) in sim_env, and record all observations.
    Save this episode of data, and continue to next episode of data collection.
    """

    task_config = SIM_TASK_CONFIGS.get(args.task_name, {}).copy()
    dataset_dir = args.dataset_dir if args.dataset_dir else task_config.get('dataset_dir')
    num_episodes = args.num_episodes if args.num_episodes is not None else task_config.get('num_episodes')
    episode_len = args.episode_len if args.episode_len is not None else task_config.get('episode_len')
    camera_list = args.camera_names.split(",") if args.camera_names else task_config.get('camera_names')
    onscreen_render = args.onscreen_render if args.onscreen_render else task_config.get('onscreen_render')
    inject_noise = args.inject_noise if args.inject_noise else task_config.get('inject_noise')

    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)

    policy_cls = PickAndTransferPolicy

    success = []
    for episode_idx in range(num_episodes):
        print(f'{episode_idx=}')
        print('Rollout out EE space scripted policy')
        # setup the environment
        env = make_sim_env(
            TransferCubeEETask, 
            'trossen_ai_scene.xml', 
            args.task_name, 
            onscreen_render=onscreen_render, 
            camera_list=camera_list
        )
        ts = env.reset()
        episode = [ts]
        policy = policy_cls(inject_noise)
        # setup plotting
        if onscreen_render:
            plt_imgs = plot_observation_images(ts.observation, camera_list)
        for step in tqdm(range(episode_len)):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_imgs = set_observation_images(ts.observation, plt_imgs, camera_list)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        if episode_max_reward == env.task.max_reward:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")

        joint_traj = [ts.observation['qpos'] for ts in episode]
        # replace gripper pose with gripper control
        # gripper_ctrl_traj = [ts.observation['gripper_ctrl'] for ts in episode]
        # for joint, ctrl in zip(joint_traj, gripper_ctrl_traj):
        #     left_ctrl = FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(ctrl[0])
        #     right_ctrl = FOLLOWER_GRIPPER_POSITION_NORMALIZE_FN(ctrl[2])
        #     joint[6] = left_ctrl
        #     joint[6+7] = right_ctrl

        subtask_info = episode[0].observation['env_state'].copy() # box pose at step 0
        print(f"Box Pose at step 0: {subtask_info}, Box Pose at step 1: {episode[1].observation['env_state']}")
        # clear unused variables
        del env
        del episode
        del policy

        # setup the environment
        print('Replaying joint commands')
        env = make_sim_env(
            TransferCubeTask, 
            'trossen_ai_scene_joint.xml',
            camera_list=camera_list
        )
        BOX_POSE[0] = subtask_info # make sure the sim_env has the same object configurations as ee_sim_env
        ts = env.reset()
        episode_replay = [ts]
        # setup plotting
        plt_imgs = plot_observation_images(ts.observation, camera_list)

        for t in tqdm(range(len(joint_traj))): # note: this will increase episode length by 1
            action = joint_traj[t]
            ts = env.step(action)
            episode_replay.append(ts)
            if onscreen_render:
                plt_imgs = set_observation_images(ts.observation, plt_imgs, camera_list)
        episode_return = np.sum([ts.reward for ts in episode_replay[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode_replay[1:]])
        if episode_max_reward == env.task.max_reward:
            success.append(1)
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            success.append(0)
            print(f"{episode_idx=} Failed")

        plt.close()

        """
        For each timestep:
        observations
        - images
            - each_cam_name     (480, 640, 3) 'uint8'
        - qpos                  (14,)         'float64'
        - qvel                  (14,)         'float64'

        action                  (14,)         'float64'
        """

        data_dict = {
            '/observations/qpos': [],
            '/observations/qvel': [],
            '/action': [],
        }
        for cam_name in camera_list:
            data_dict[f'/observations/images/{cam_name}'] = []

        # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
        # truncate here to be consistent
        joint_traj = joint_traj[:-1]
        episode_replay = episode_replay[:-1]

        # len(joint_traj) i.e. actions: max_timesteps
        # len(episode_replay) i.e. time steps: max_timesteps + 1
        max_timesteps = len(joint_traj)
        while joint_traj:
            action = joint_traj.pop(0)
            ts = episode_replay.pop(0)
            data_dict['/observations/qpos'].append(ts.observation['qpos'])
            data_dict['/observations/qvel'].append(ts.observation['qvel'])
            data_dict['/action'].append(action)
            for cam_name in camera_list:
                data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])

        # HDF5
        t0 = time.time()
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
        with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
            root.attrs['sim'] = True
            obs = root.create_group('observations')
            image = obs.create_group('images')
            for cam_name in camera_list:
                _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                         chunks=(1, 480, 640, 3), )
            # compression='gzip',compression_opts=2,)
            # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
            qpos = obs.create_dataset('qpos', (max_timesteps, 16))
            qvel = obs.create_dataset('qvel', (max_timesteps, 16))
            action = root.create_dataset('action', (max_timesteps, 16))

            for name, array in data_dict.items():
                root[name][...] = array
        print(f'Saving: {time.time() - t0:.1f} secs\n')

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Record simulation episodes with customization.")
    parser.add_argument('--task_name', type=str, default="sim_transfer_cube", help="Name of the task.")
    parser.add_argument('--dataset_dir', type=str, help="Directory to save episodes.")
    parser.add_argument('--num_episodes', type=int, help="Number of episodes to run.")
    parser.add_argument('--episode_len', type=int, help="Length of each episode.")
    parser.add_argument('--onscreen_render', action='store_true', help="Enable on-screen rendering.")
    parser.add_argument('--inject_noise', action='store_true', help="Inject noise into actions.")
    parser.add_argument('--camera_names', type=str, help="Comma-separated list of camera names.")

    args = parser.parse_args()
    main(args)
