import os
import h5py
import cv2
import numpy as np
import argparse


def load_hdf5(dataset_path):
    """
    Load the camera feeds from an HDF5 dataset.
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at {dataset_path}")
        exit()

    with h5py.File(dataset_path, 'r') as root:
        image_dict = {}
        for cam_name in root['/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return image_dict


def save_videos(image_dict, dt, video_path):
    """
    Save all camera feeds into a single video file.
    """
    cam_names = list(image_dict.keys())
    h, w, _ = image_dict[cam_names[0]][0].shape
    w_total = w * len(cam_names)  # Concatenate all cameras horizontally
    fps = int(1 / dt)

    # Initialize video writer
    out = cv2.VideoWriter(
        video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w_total, h)
    )

    num_frames = len(image_dict[cam_names[0]])

    for frame_idx in range(num_frames):
        frame_row = []
        for cam_name in cam_names:
            image = image_dict[cam_name][frame_idx]
            image = image[:, :, [2, 1, 0]]  # Convert BGR to RGB
            frame_row.append(image)

        concatenated_frame = np.concatenate(frame_row, axis=1)  # Horizontally concatenate
        out.write(concatenated_frame)

    out.release()
    print(f"Saved video to: {video_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert camera feeds in HDF5 to MP4 video.")
    parser.add_argument(
        '--dataset_path',
        required=True,
        help='Path to the HDF5 dataset file.'
    )
    parser.add_argument(
        '--output_path',
        required=True,
        help='Path to save the output MP4 video file.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=50,
        help='Frames per second for the video. Default is 50.'
    )

    args = parser.parse_args()

    # Load camera data from HDF5
    images = load_hdf5(args.dataset_path)

    # Save to MP4 video
    save_videos(images, dt=1/args.fps, video_path=args.output_path)
