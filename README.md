# Trossen Arm MuJoCo

## Overview

This package provides the necessary scripts and assets for simulating and training robotic policies using the Trossen AI kits in MuJoCo.
It includes URDFs, mesh models, and MuJoCo XML files for robot configuration, as well as Python scripts for policy execution, reward-based evaluation, data collection, and visualization.

This package supports two types of simulation environments:

1. End-Effector (EE) Controlled Simulation ([`ee_sim_env.py`](./trossen_arm_mujoco/ee_sim_env.py)): Uses motion capture bodies to move the arms
2. Joint-Controlled Simulation ([`sim_env.py`](./trossen_arm_mujoco/sim_env.py)): Uses joint position controllers

## Installation

First, clone this repository:

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
```

It is recommended to create a virtual environment before installing dependencies.
Create a Conda environment with Python 3.10 or above.

```bash
conda create --name trossen_mujoco_env python=3.10
```

After creation, activate the environment with:

```bash
conda activate trossen_mujoco_env
```

Install the package and required dependencies using:

```bash
cd trossen_arm_mujoco
pip install .
```

To verify the installation, run:

```bash
python trossen_arm_mujoco/ee_sim_env.py
```

If the simulation window appears, the setup was successful.

## 1. Assets ([`assets/`](./trossen_arm_mujoco/assets/))

This folder contains all required MuJoCo XML configuration files, URDF files, and mesh models for the simulation.

### Key Files:

- `trossen_ai.xml` → Base model definition of the Trossen AI robot.
- `trossen_ai_scene.xml` → Uses mocap bodies to control the simulated arms.
- `trossen_ai_scene_joint.xml` → Uses joint controllers similar to real hardware to control the simulated arms.
- `wxai_follower.urdf` & `wxai_follower.xml` → URDF and XML descriptions of the follower arms.
- `meshes/` → Contains STL and OBJ files for the robot components, including arms, cameras, and environmental objects.

### Motion Capture vs Joint-Controlled Environments:

- Motion Capture (`trossen_ai_scene.xml`): Uses predefined mocap bodies that move the robot arms based on scripted end effector movements.
- Joint Control (`trossen_ai_scene_joint.xml`): Uses position controllers for each joint, similar to a real-world robot setup.

## 2. Modules ([`trossen_arm_mujoco`](./trossen_arm_mujoco/))

This folder contains all Python modules necessary for running simulations, executing policies, recording episodes, and visualizing results.

### 2.1 Simulations

- `ee_sim_env.py`
  - Loads `trossen_ai_scene.xml` (motion capture-based control).
  - The arms move by following the positions commanded to the mocap bodies.
  - Used for generating scripted policies that control the robot’s arms in predefined ways.

- `sim_env.py`
  - Loads `trossen_ai_scene_joint.xml` (position-controlled joints).
  - Uses joint controllers instead of mocap bodies.
  - Mimics the real robot’s movement with controlled joint actuation.

### 2.2 Scripted Policy Execution

- `scripted_policy.py`
  - Defines pre-scripted movements for the robot arms to perform tasks like picking up objects.
  - Uses the motion capture bodies to generate smooth movement trajectories.
  - In the current setup, a policy is designed to pick up a red block, with randomized block positions in the environment.

## 3. How the Data Collection Works

The data collection process involves two simulation phases:

1. Running a scripted policy in `ee_sim_env.py` to record observations (joint positions).
2. Replaying the recorded joint positions in `sim_env.py` to capture full episode data.

### Step-by-Step Process

1. Run `record_sim_episodes.py`

   - Starts `ee_sim_env.py` and executes a scripted policy.
   - Captures observations in the form of joint positions.
   - Saves these joint positions for later replay.

2. Replay in `sim_env.py`

   - Uses the previously recorded joint positions as input commands.
   - Replays the episode in the joint-controlled simulation (`sim_env.py`).
   - Captures additional data, including:
     - Camera feeds from 4 different viewpoints
     - Joint states (actual positions during execution)
     - Actions (input joint positions)
     - Reward values to determine success or failure

3. Save the Data

   - All observations and actions are stored in HDF5 format, with one file per episode.
   - Each episode is saved as `episode_X.hdf5` inside the `~/.trossen/mujoco/data/` folder.

4. Visualizing the Data

   - The stored HDF5 files can be converted into videos using `visualize.py`.
   - The resulting videos are saved in MP4 format inside the `ee_sim_episodes_output/` folder.

## 4. Script Arguments Explanation

### a. record_sim_episodes.py

This script generates and saves demonstration episodes using a scripted policy in simulation. It supports both end-effector control (for task definition) and joint-space replay (for clean data collection), storing all observations in `.hdf5` format.

To generate and save simulation episodes, use:

```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --num_episodes 5 \
    --data_dir data/sim_transfer_cube \
    --onscreen_render
```
Arguments:

- `--task_name`: Name of the task (default: sim_transfer_cube).
- `--num_episodes`: Number of episodes to generate.
- `--data_dir`: Directory where episodes will be saved. Default: `data/sim_transfer_cube`
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`
- `--episode_len`: Length of each episode.
- `--onscreen_render` : Enables on-screen rendering. Default: False (only true if explicitly set)
- `--inject_noise`: Injects noise into actions. Default: False (only true if explicitly set)
- `--cam_names`: Comma-separated list of camera names for image collection

**Note:**

- When you pass `--task_name`, the script will automatically load the corresponding configuration from constants.py.

- You can extend `SIM_TASK_CONFIGS` in `constants.py` to support new task configurations.

- All parameters loaded from `constants.py` can be individually overridden via command-line arguments.

### b. visualize_obs.py

To convert saved episodes to videos, run:

```bash
python trossen_arm_mujoco/scripts/visualize_obs.py \
    --data_dir data/sim_transfer_cube \
    --output_dir data/videos \
    --fps 50
```
Arguments:

- `--data_dir`: Directory containing `.hdf5` files. Default: `data/sim_transfer_cube`
- `--output_dir`: Output directory for `.mp4` files. Default: `videos`
- `--fps`: Frames per second for the generated videos (default: 50)
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`

### c. replay_episode.py

This script replays recorded joint-space episodes on real Trossen robotic arms using data saved in .hdf5 files. 
It configures each arm, plays back the actions with a user-defined frame rate, and returns both arms to a safe rest pose after execution.

To perform sim to real, run:

```bash
python trossen_arm_mujoco/scripts/replay_episode.py \
    --data_dir data/sim_transfer_cube \
    --episode_idx 0 \
    --fps 10 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

Arguments:

- `--data_dir`: Directory containing `.hdf5` files
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`
- `--episode_idx`: Index of the episode to replay. Default: 0
- `--fps`: Playback frame rate (Hz). Controls the action replay speed. Default: 10
- `--left_ip` : IP address of the left Trossen arm. Default: 192.168.1.5
- `--right_ip`: 	IP address of the right Trossen arm. Default: 192.168.1.4

## Customization

### 1. Modifying Tasks

To create a custom task, modify `ee_sim_env.py` or `sim_env.py` and define a new subclass of `TrossenAIStationary(EE)Task`.
Implement:

- `initialize_episode(self, physics)`: Set up the initial environment state, including robot and object positions.
- `get_observation(self, physics)`: Define what data should be recorded as observations.
- `get_reward(self, physics)`: Implement the reward function to determine task success criteria.

### 2. Changing Policy Behavior

Modify `scripted_policy.py` to define new behavior for the robotic arms.
Update the trajectory generation logic in `PickAndTransferPolicy.generate_trajectory()` to create different movement patterns.

Each movement step in the trajectory is defined by:

- `t`: The time step at which the movement shall occur.
- `xyz`: The target position of the end effector in 3D space.
- `quat`: The target orientation of the end effector, represented as a quaternion.
- `gripper`: The target gripper finger position 0~0.065 where 0 is closed and 0.065 is fully open.

Example:

```python
def generate_trajectory(self, ts_first: TimeStep):
    self.left_trajectory = [
        {"t": 0, "xyz": [0, 0, 0.4], "quat": [1, 0, 0, 0], "gripper": 0},
        {"t": 100, "xyz": [0.1, 0, 0.3], "quat": [1, 0, 0, 0], "gripper": 0.05}
    ]
```

### 3. Adding New Environment Setups

The simulation uses XML files stored in the `assets/` directory. To introduce a new environment setup:

1. Create a new XML configuration file in `assets/` with desired object placements and constraints.
2. Modify `sim_env.py` to load the new environment by specifying the new XML file.
3. Update the scripted policies in `scripted_policy.py` to accommodate new task goals and constraints.

## Troubleshooting

If you encounter into Mesa Loader or `mujoco.FatalError: gladLoadGL error` errors:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```
