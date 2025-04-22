import h5py
import numpy as np
import csv
import os

def extract_to_csv_with_timestamp(hdf5_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    with h5py.File(hdf5_path, 'r') as f:
        if 'observations' not in f or 'action' not in f:
            raise KeyError("Expected groups 'observations' and 'action' not found in the file.")

        observations = f['observations']
        qpos = observations['qpos'][:]   # Shape: (T, 14)
        qvel = observations['qvel'][:]   # Shape: (T, 14)
        actions = f['action'][:]         # Shape: (T, 14)
        timesteps = np.arange(qpos.shape[0])

        # Write qpos with timestep
        with open(os.path.join(output_dir, 'qpos.csv'), 'w', newline='') as f_qpos:
            writer = csv.writer(f_qpos)
            header = ['timestep'] + [f'qpos_{i}' for i in range(qpos.shape[1])]
            writer.writerow(header)
            for t, row in zip(timesteps, qpos):
                writer.writerow([t] + list(row))

        # Write qvel with timestep
        with open(os.path.join(output_dir, 'qvel.csv'), 'w', newline='') as f_qvel:
            writer = csv.writer(f_qvel)
            header = ['timestep'] + [f'qvel_{i}' for i in range(qvel.shape[1])]
            writer.writerow(header)
            for t, row in zip(timesteps, qvel):
                writer.writerow([t] + list(row))

        # Write actions with timestep
        with open(os.path.join(output_dir, 'action.csv'), 'w', newline='') as f_action:
            writer = csv.writer(f_action)
            header = ['timestep'] + [f'action_{i}' for i in range(actions.shape[1])]
            writer.writerow(header)
            for t, row in zip(timesteps, actions):
                writer.writerow([t] + list(row))

    print(f"Data saved to: {output_dir}")

extract_to_csv_with_timestamp("/home/shuhang/workspace/MuJoCo/original/trossen_arm_mj/aloha_data/ee_sim_episodes_10/episode_0.hdf5", "output_csvs")
