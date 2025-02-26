# import cv2
# import fnmatch
import numpy as np
# import torch
import os
import collections
import numpy as np
from dm_control import mujoco
from dm_control.rl import control
import importlib
import matplotlib.pyplot as plt
# import h5py
# from torch.utils.data import TensorDataset, DataLoader


# class EpisodicDataset(torch.utils.data.Dataset):
#     def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats):
#         super(EpisodicDataset).__init__()
#         self.episode_ids = episode_ids
#         self.dataset_dir = dataset_dir
#         self.camera_names = camera_names
#         self.norm_stats = norm_stats
#         self.is_sim = None
#         self.__getitem__(0) # initialize self.is_sim

#     def __len__(self):
#         return len(self.episode_ids)

#     def __getitem__(self, index):
#         sample_full_episode = False # hardcode

#         episode_id = self.episode_ids[index]
#         dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             is_sim = root.attrs['sim']
#             compressed = root.attrs['compress']
#             original_action_shape = root['/action'].shape
#             episode_len = original_action_shape[0]
#             if sample_full_episode:
#                 start_ts = 0
#             else:
#                 start_ts = np.random.choice(episode_len)
#             # get observation at start_ts only
#             qpos = root['/observations/qpos'][start_ts]
#             qvel = root['/observations/qvel'][start_ts]
#             image_dict = dict()
#             for cam_name in self.camera_names:
#                 image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

#             # Decompress the images to match the shape of tensor required for torch.einsum()
#             if compressed:
#                 for cam_name in image_dict.keys():
#                     decompressed_image = cv2.imdecode(image_dict[cam_name],1)
#                     image_dict[cam_name] = np.array(decompressed_image)

#             # get all actions after and including start_ts
#             if is_sim:
#                 action = root['/action'][start_ts:]
#                 action_len = episode_len - start_ts
#             else:
#                 # hack, to make timesteps more aligned
#                 action = root['/action'][max(0, start_ts - 1):]
#                 # hack, to make timesteps more aligned
#                 action_len = episode_len - max(0, start_ts - 1)

#         self.is_sim = is_sim
#         padded_action = np.zeros(original_action_shape, dtype=np.float32)
#         padded_action[:action_len] = action
#         is_pad = np.zeros(episode_len)
#         is_pad[action_len:] = 1

#         # new axis for different cameras
#         all_cam_images = []
#         for cam_name in self.camera_names:
#             all_cam_images.append(image_dict[cam_name])
#         all_cam_images = np.stack(all_cam_images, axis=0)

#         # construct observations
#         image_data = torch.from_numpy(all_cam_images)
#         qpos_data = torch.from_numpy(qpos).float()
#         action_data = torch.from_numpy(padded_action).float()
#         is_pad = torch.from_numpy(is_pad).bool()

#         # channel last
#         image_data = torch.einsum('k h w c -> k c h w', image_data)

#         # normalize image and change dtype to float
#         image_data = image_data / 255.0
#         action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
#         qpos_data = (qpos_data - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]

#         return image_data, qpos_data, action_data, is_pad


# def get_norm_stats(dataset_dir, num_episodes):
#     all_qpos_data = []
#     all_action_data = []
#     for episode_idx in range(num_episodes):
#         dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}.hdf5')
#         with h5py.File(dataset_path, 'r') as root:
#             qpos = root['/observations/qpos'][()]
#             qvel = root['/observations/qvel'][()]
#             action = root['/action'][()]
#         all_qpos_data.append(torch.from_numpy(qpos))
#         all_action_data.append(torch.from_numpy(action))
#     all_qpos_data = torch.stack(all_qpos_data)
#     all_action_data = torch.stack(all_action_data)
#     all_action_data = all_action_data

#     # normalize action data
#     action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)
#     action_std = all_action_data.std(dim=[0, 1], keepdim=True)
#     action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

#     # normalize qpos data
#     qpos_mean = all_qpos_data.mean(dim=[0, 1], keepdim=True)
#     qpos_std = all_qpos_data.std(dim=[0, 1], keepdim=True)
#     qpos_std = torch.clip(qpos_std, 1e-2, np.inf) # clipping

#     stats = {
#         "action_mean": action_mean.numpy().squeeze(),
#         "action_std": action_std.numpy().squeeze(),
#         "qpos_mean": qpos_mean.numpy().squeeze(),
#         "qpos_std": qpos_std.numpy().squeeze(),
#         "example_qpos": qpos,
#     }

#     return stats


# def find_all_hdf5(dataset_dir, skip_mirrored_data):
#     hdf5_files = []
#     for root, dirs, files in os.walk(dataset_dir):
#         for filename in fnmatch.filter(files, '*.hdf5'):
#             if 'features' in filename:
#                 continue
#             if skip_mirrored_data and 'mirror' in filename:
#                 continue
#             hdf5_files.append(os.path.join(root, filename))
#     print(f'Found {len(hdf5_files)} hdf5 files')
#     return hdf5_files


# def load_data(dataset_dir, camera_names, batch_size_train, batch_size_val, skip_mirrored_data=False):
#     print(f'\nData from: {dataset_dir}\n')

#     # verify that the directory passed is a string
#     if isinstance(dataset_dir, str):
#         # get all the episodes from the directory.
#         dataset_path_list_list = [find_all_hdf5(dataset_dir, skip_mirrored_data)]
#         # get the length of the list. Store it as number of episodes.
#         num_episodes = len(dataset_path_list_list[0])

#     # obtain train test split
#     train_ratio = 0.8
#     shuffled_indices = np.random.permutation(num_episodes)
#     train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
#     val_indices = shuffled_indices[int(train_ratio * num_episodes):]

#     # obtain normalization stats for qpos and action
#     norm_stats = get_norm_stats(dataset_dir, num_episodes)

#     # construct dataset and dataloader
#     train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names, norm_stats)
#     val_dataset = EpisodicDataset(val_indices, dataset_dir, camera_names, norm_stats)
#     train_dataloader = DataLoader(
#         train_dataset,
#         batch_size=batch_size_train,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=1,
#         prefetch_factor=1,
#     )
#     val_dataloader = DataLoader(
#         val_dataset,
#         batch_size=batch_size_val,
#         shuffle=True,
#         pin_memory=True,
#         num_workers=1,
#         prefetch_factor=1,
#     )

#     return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


# ### env utils

def sample_box_pose():
    x_range = [-0.1, 0.2]
    y_range = [-0.15, 0.15]
    z_range = [0.02, 0.02]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    print(f"Cube Position: {cube_position}")
    return np.concatenate([cube_position, cube_quat])

def get_observation_base(physics, on_screen_render=True):
    obs = collections.OrderedDict()
    if on_screen_render:
        obs["images"] = dict()
        obs["images"]["camera_high"] = physics.render(height=480, width=640, camera_id="camera_high")
        obs["images"]["camera_low"] = physics.render(height=480, width=640, camera_id="camera_low")
        obs["images"]["camera_left_wrist"] = physics.render(height=480, width=640, camera_id="camera_left_wrist")
        obs["images"]["camera_right_wrist"] = physics.render(height=480, width=640, camera_id="camera_right_wrist")
        obs["images"]["camera_teleop"] = physics.render(height=480, width=640, camera_id="teleoperator_pov")
    return obs

XML_DIR = "assets"
DT = 0.02

def make_sim_env(task_class, xml_file='aloha_scene.xml', task_name='sim_transfer_cube', onscreen_render=False):
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
        task = task_class(random=False, onscreen_render=onscreen_render)
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

def plot_observation_images(observation):
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    plt_imgs = [
        axs[0, 0].imshow(observation['images']['camera_high']),
        axs[0, 1].imshow(observation['images']['camera_low']),
        axs[1, 0].imshow(observation['images']['camera_left_wrist']),
        axs[1, 1].imshow(observation['images']['camera_right_wrist']),
    ]

    # Optionally, add titles for better clarity
    axs[0, 0].set_title("Camera High")
    axs[0, 1].set_title("Camera Low")
    axs[1, 0].set_title("Left Wrist Camera")
    axs[1, 1].set_title("Right Wrist Camera")


    # Remove axis ticks for better visualization
    for ax in axs.flat:
        ax.axis('off')

    # plt.tight_layout()
    plt.ion()

# def sample_insertion_pose():
#     # Peg
#     x_range = [0.1, 0.2]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     peg_quat = np.array([1, 0, 0, 0])
#     peg_pose = np.concatenate([peg_position, peg_quat])

#     # Socket
#     x_range = [-0.2, -0.1]
#     y_range = [0.4, 0.6]
#     z_range = [0.05, 0.05]

#     ranges = np.vstack([x_range, y_range, z_range])
#     socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

#     socket_quat = np.array([1, 0, 0, 0])
#     socket_pose = np.concatenate([socket_position, socket_quat])

#     return peg_pose, socket_pose

# ### helper functions

# def compute_dict_mean(epoch_dicts):
#     result = {k: None for k in epoch_dicts[0]}
#     num_items = len(epoch_dicts)
#     for k in result:
#         value_sum = 0
#         for epoch_dict in epoch_dicts:
#             value_sum += epoch_dict[k]
#         result[k] = value_sum / num_items
#     return result


# def detach_dict(d):
#     new_d = dict()
#     for k, v in d.items():
#         new_d[k] = v.detach()
#     return new_d


# def set_seed(seed):
#     torch.manual_seed(seed)
#     np.random.seed(seed)


# def save_videos(video, dt, video_path=None):
#     if isinstance(video, list):
#         cam_names = list(video[0].keys())
#         h, w, _ = video[0][cam_names[0]].shape
#         w = w * len(cam_names)
#         fps = int(1/dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for ts, image_dict in enumerate(video):
#             images = []
#             for cam_name in cam_names:
#                 image = image_dict[cam_name]
#                 image = image[:, :, [2, 1, 0]] # swap B and R channel
#                 images.append(image)
#             images = np.concatenate(images, axis=1)
#             out.write(images)
#         out.release()
#         print(f'Saved video to: {video_path}')
        
#     elif isinstance(video, dict):
#         cam_names = list(video.keys())
#         all_cam_videos = []
#         for cam_name in cam_names:
#             all_cam_videos.append(video[cam_name])
#         all_cam_videos = np.concatenate(all_cam_videos, axis=2) # width dimension

#         n_frames, h, w, _ = all_cam_videos.shape
#         fps = int(1 / dt)
#         out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
#         for t in range(n_frames):
#             image = all_cam_videos[t]
#             image = image[:, :, [2, 1, 0]]  # swap B and R channel
#             out.write(image)
#         out.release()
#         print(f'Saved video to: {video_path}')
