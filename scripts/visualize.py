import os
import h5py
import cv2
import numpy as np
import argparse
import re

def load_hdf5(dataset_path):
    """
    Load the camera feeds from an HDF5 dataset.
    """
    if not os.path.isfile(dataset_path):
        print(f"Dataset does not exist at {dataset_path}")
        return None

    with h5py.File(dataset_path, 'r') as root:
        image_dict = {}
        for cam_name in root['/observations/images/'].keys():
            image_dict[cam_name] = root[f'/observations/images/{cam_name}'][()]

    return image_dict


def save_videos(image_dict, dt, video_path):
    """
    Save all camera feeds into a single video file.
    """
    if not image_dict:
        print(f"Skipping {video_path}: No valid images found.")
        return
    
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
        frame_row = [image_dict[cam_name][frame_idx][:, :, [2, 1, 0]] for cam_name in cam_names]  # Convert RGB to BGR
        concatenated_frame = np.concatenate(frame_row, axis=1)  # Horizontally concatenate
        out.write(concatenated_frame)

    out.release()
    print(f"Saved video to: {video_path}")


def process_directory(dataset_dir, output_dir, fps=50):
    """
    Convert all episodes in the dataset directory to videos.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Match episode files (episode_{number}.hdf5)
    episode_pattern = re.compile(r'episode_(\d+)\.hdf5')

    # Iterate over all files in the directory
    for filename in os.listdir(dataset_dir):
        match = episode_pattern.match(filename)
        if match:
            episode_number = match.group(1)
            input_path = os.path.join(dataset_dir, filename)
            print(f"Processing {input_path}")
            output_path = os.path.join(output_dir, f"episode_{episode_number}.mp4")

            print(f"Processing {input_path} → {output_path}")

            # Load camera data
            images = load_hdf5(input_path)
            if images:
                # Save to MP4 video
                save_videos(images, dt=1/fps, video_path=output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert all HDF5 episodes in a directory to MP4 videos.")
    parser.add_argument(
        '--dataset_dir',
        required=True,
        help='Path to the directory containing HDF5 dataset files.'
    )
    parser.add_argument(
        '--output_dir',
        required=True,
        help='Path to the directory to save the output MP4 videos.'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=50,
        help='Frames per second for the video. Default is 50.'
    )

    args = parser.parse_args()

    # Process all episodes in the directory
    process_directory(args.dataset_dir, args.output_dir, fps=args.fps)
