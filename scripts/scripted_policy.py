import numpy as np
import argparse
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from constants import SIM_TASK_CONFIGS
from utils import make_sim_env, plot_observation_images, set_observation_images
import IPython
from ee_sim_env import TransferCubeEETask

class BasePolicy:
    def __init__(self, inject_noise=False):
        self.inject_noise = inject_noise
        self.step_count = 0
        self.left_trajectory = None
        self.right_trajectory = None

    def generate_trajectory(self, ts_first):
        raise NotImplementedError

    @staticmethod
    def interpolate(curr_waypoint, next_waypoint, t):
        t_frac = (t - curr_waypoint["t"]) / (next_waypoint["t"] - curr_waypoint["t"])
        curr_xyz = curr_waypoint['xyz']
        curr_quat = curr_waypoint['quat']
        curr_grip = curr_waypoint['gripper']
        next_xyz = next_waypoint['xyz']
        next_quat = next_waypoint['quat']
        next_grip = next_waypoint['gripper']
        xyz = curr_xyz + (next_xyz - curr_xyz) * t_frac
        quat = curr_quat + (next_quat - curr_quat) * t_frac
        gripper = curr_grip + (next_grip - curr_grip) * t_frac
        return xyz, quat, gripper

    def __call__(self, ts):
        # generate trajectory at first timestep, then open-loop execution
        if self.step_count == 0:
            self.generate_trajectory(ts)

        # obtain left and right waypoints
        if self.left_trajectory[0]['t'] == self.step_count:
            self.curr_left_waypoint = self.left_trajectory.pop(0)
        next_left_waypoint = self.left_trajectory[0]

        if self.right_trajectory[0]['t'] == self.step_count:
            self.curr_right_waypoint = self.right_trajectory.pop(0)
        next_right_waypoint = self.right_trajectory[0]

        # interpolate between waypoints to obtain current pose and gripper command
        left_xyz, left_quat, left_gripper = self.interpolate(self.curr_left_waypoint, next_left_waypoint, self.step_count)
        right_xyz, right_quat, right_gripper = self.interpolate(self.curr_right_waypoint, next_right_waypoint, self.step_count)

        # Inject noise
        if self.inject_noise:
            scale = 0.01
            left_xyz = left_xyz + np.random.uniform(-scale, scale, left_xyz.shape)
            right_xyz = right_xyz + np.random.uniform(-scale, scale, right_xyz.shape)

        action_left = np.concatenate([left_xyz, left_quat, [left_gripper]])
        action_right = np.concatenate([right_xyz, right_quat, [right_gripper]])

        self.step_count += 1
        return np.concatenate([action_left, action_right])


class PickAndTransferPolicy(BasePolicy):

    def generate_trajectory(self, ts_first):
        init_mocap_pose_right = ts_first.observation['mocap_pose_right']
        init_mocap_pose_left = ts_first.observation['mocap_pose_left']

        box_info = np.array(ts_first.observation['env_state'])
        box_xyz = box_info[:3]
        print(f"Generate trajectory for {box_xyz=}")
        box_quat = box_info[3:]
        # print(f"Generate trajectory for {box_xyz=}")

        gripper_pick_quat = Quaternion(init_mocap_pose_right[3:])
        gripper_pick_quat = gripper_pick_quat * Quaternion(axis=[0.0, 1.0, 0.0], degrees=-60)

        meet_left_quat = Quaternion(axis=[1.0, 0.0, 0.0], degrees=90)

        meet_xyz = np.array([0.0, 0.0, 0.25])

        self.left_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_left[:3], "quat": init_mocap_pose_left[3:], "gripper": 0}, # sleep
            {"t": 100, "xyz": meet_xyz + np.array([-0.3, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0.06}, # approach meet position
            {"t": 210, "xyz": meet_xyz + np.array([-0.2, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0.065}, # move to meet position
            {"t": 250, "xyz": meet_xyz + np.array([-0.13, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0.065}, # move to meet position
            {"t": 310, "xyz": meet_xyz + np.array([-0.12, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0.02}, # close gripper
            {"t": 360, "xyz": meet_xyz + np.array([-0.2, 0, -0.02]), "quat": meet_left_quat.elements, "gripper": 0.02}, # move left
            {"t": 400, "xyz": meet_xyz + np.array([-0.2, 0, -0.02]), "quat": np.array([1, 0, 0, 0]), "gripper": 0.02}, # stay
        ]

        self.right_trajectory = [
            {"t": 0, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0}, # sleep
            {"t": 5, "xyz": init_mocap_pose_right[:3], "quat": init_mocap_pose_right[3:], "gripper": 0.05}, # open gripper
            {"t": 90, "xyz": box_xyz + np.array([0.1, 0, 0.2]), "quat": gripper_pick_quat.elements, "gripper": 0.055}, # approach the cube
            {"t": 130, "xyz": box_xyz + np.array([0.06, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0.055}, # go down
            {"t": 140, "xyz": box_xyz + np.array([0.06, 0, 0.12]), "quat": gripper_pick_quat.elements, "gripper": 0.02}, # close gripper
            {"t": 200, "xyz": meet_xyz + np.array([0.08, 0, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0.02}, # approach meet position
            {"t": 320, "xyz": meet_xyz + np.array([0.08, 0, 0.1]),  "quat": gripper_pick_quat.elements, "gripper": 0.02}, # move to meet position
            {"t": 350, "xyz": meet_xyz + np.array([0.08, 0, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0.065}, # open gripper
            {"t": 360, "xyz": meet_xyz + np.array([0.2, 0, 0.15]), "quat": gripper_pick_quat.elements, "gripper": 0.065}, # move to right
            {"t": 400, "xyz": meet_xyz + np.array([0.2, 0, 0.1]), "quat": gripper_pick_quat.elements, "gripper": 0.065}, # stay
        ]



def test_policy(task_name, num_episodes=2, episode_len=400, onscreen_render=True, inject_noise=False):
    # setup the environment
    camera_list = ["camera_high", "camera_low", "camera_left_wrist", "camera_right_wrist"]
    env = make_sim_env(TransferCubeEETask, task_name=task_name, onscreen_render=onscreen_render, camera_list=camera_list)
    print(f"Action space: {env.action_spec().shape}")

    for episode_idx in range(num_episodes):
        ts = env.reset()
        episode = [ts]
        if onscreen_render:
            plt_imgs = plot_observation_images(ts.observation, camera_list)

        policy = PickAndTransferPolicy(inject_noise)
        for step in range(episode_len):
            action = policy(ts)
            ts = env.step(action)
            episode.append(ts)
            if onscreen_render:
                plt_imgs = set_observation_images(ts.observation, plt_imgs, camera_list)
        plt.close()

        episode_return = np.sum([ts.reward for ts in episode[1:]])
        if episode_return > 0:
            print(f"{episode_idx=} Successful, {episode_return=}")
        else:
            print(f"{episode_idx=} Failed")


if __name__ == '__main__':
    # test_task_name = 'sim_transfer_cube_scripted'
    parser = argparse.ArgumentParser(description="Test policy with customizable parameters.")
    parser.add_argument('--task_name', type=str, default="sim_transfer_cube", help="Task name.")
    parser.add_argument('--num_episodes', type=int, help="Number of episodes.")
    parser.add_argument('--episode_len', type=int, help="Episode length.")
    parser.add_argument('--onscreen_render', action='store_true', help="Enable rendering.")
    parser.add_argument('--inject_noise', action='store_true', help="Inject noise into actions.")

    args = parser.parse_args()
    task_config = SIM_TASK_CONFIGS.get(args.task_name, {}).copy()
    num_episodes = args.num_episodes if args.num_episodes is not None else task_config.get('num_episodes')
    episode_len = args.episode_len if args.episode_len is not None else task_config.get('episode_len')
    onscreen_render = args.onscreen_render if args.onscreen_render else task_config.get('onscreen_render')
    inject_noise = args.inject_noise if args.inject_noise else task_config.get('inject_noise')
    test_policy(args.task_name, num_episodes, episode_len, onscreen_render, inject_noise)

