# Trossen Arm MuJoCo

[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.2.3-orange.svg)](https://mujoco.readthedocs.io/) [![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html) [![Linux](https://img.shields.io/badge/platform-Ubuntu_22.04-lightgrey.svg)](https://releases.ubuntu.com/22.04/)

## Overview

This repository provides MuJoCo simulation environments for Trossen AI robotic arms. It includes comprehensive robot models, inverse kinematics-based control, and tools for policy execution, data collection, and visualization.

### What This Repository Offers

- MuJoCo XML models for Trossen AI robots:
  - WidowX AI (single arm manipulator)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Differential inverse kinematics controller for Cartesian end-effector control
- Pick-and-place and target following demonstration scripts
- Motion capture and joint-controlled simulation environments
- Data collection and visualization tools for imitation learning

### Tested Environment

- Ubuntu 22.04
- MuJoCo 3.2.3
- Python 3.10+

---

## Index

- [Overview](#overview)
- [Robot Assets](#robot-assets)
- [Installation](#installation)
- [MuJoCo IK Demo Scripts](#mujoco-ik-demo-scripts)
- [Controller API](#controller-api)
- [Data Collection](#data-collection)
- [Data Collection Workflow](#data-collection-workflow)
- [Script Arguments](#script-arguments)
- [Customization](#customization)
- [Troubleshooting](#troubleshooting)
- [Related Links](#related-links)

---

## Robot Assets

All robot models are located in `trossen_arm_mujoco/assets/`:

```
trossen_arm_mujoco/assets/
├── meshes/          # STL files for robot components
├── wxai/
│   ├── wxai_base.xml              # WidowX AI base model
│   ├── wxai_follower.xml          # WidowX AI follower arm with camera
│   ├── scene_wxai_pick_place.xml  # Pick-and-place scene
│   └── scene_wxai_follow_target.xml # Target following scene
├── stationary_ai/
│   ├── stationary_ai.xml          # Dual-arm stationary platform
│   ├── stationary_ai_mocap.xml    # Mocap-enabled model with weld constraints
│   ├── scene_stationary_ai_pick_place.xml # Pick-and-place with handoff
│   ├── scene_joint.xml            # Joint-controlled setup
│   └── scene_mocap.xml            # Motion capture-controlled setup
└── mobile_ai/
    ├── mobile_ai.xml              # Mobile base + dual-arm
    └── scene_mobile_ai_pick_place.xml # Mobile pick-and-place
```

### Asset Details

**WidowX AI** - Single-arm manipulator:
- Base model (`wxai_base.xml`): 6-DOF arm without camera
- Follower model (`wxai_follower.xml`): 6-DOF arm + parallel jaw gripper

**Stationary AI** - Dual-arm stationary platform:
- Dual WXAI arms on shared base
- Supports bimanual coordination and object handoff

**Mobile AI** - Mobile manipulator:
- Differential drive mobile base
- Dual WXAI arms
- Navigation and manipulation capabilities

### Motion Capture vs Joint-Controlled Environments

**Motion Capture** (`stationary_ai/scene_mocap.xml`):
- Uses predefined mocap bodies that move the robot arms based on scripted end effector movements
- Useful for defining task trajectories and generating reference motions

**Joint Control** (`stationary_ai/scene_joint.xml`):
- Uses position controllers for each joint, similar to real-world robot setup
- Used for replaying recorded trajectories with realistic joint-level control
- Enables clean simulation visuals without mocap bodies visible in rendered output

All models are derived from URDF descriptions in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description). See individual `README.md` files in asset folders for detailed URDF→MJCF derivation steps.

---

## Installation

### Clone Repository

First, clone this repository:

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
cd trossen_arm_mujoco
```

### Create Conda Environment

It is recommended to create a Conda environment before installing dependencies:

```bash
conda create --name trossen_mujoco_env python=3.10
conda activate trossen_mujoco_env
```

### Install Package

Install the package and required dependencies using:

```bash
pip install -e .
```

### Verify Installation

To verify the installation, run:

```bash
python3 trossen_arm_mujoco/scripts/wxai_pick_place.py
```

If the simulation window appears with the robot performing pick-and-place, the setup was successful.

---

## MuJoCo IK Demo Scripts

These scripts demonstrate inverse kinematics-based control for manipulation tasks.

### Pick and Place Demonstrations

**WidowX AI** - Single arm pick-and-place:
```bash
python3 trossen_arm_mujoco/scripts/wxai_pick_place.py
```

**Stationary AI** - Dual-arm pick-and-place with handoff:
```bash
python3 trossen_arm_mujoco/scripts/stationary_ai_pick_place.py
```

**Mobile AI** - Mobile base navigation + dual-arm manipulation:
```bash
python3 trossen_arm_mujoco/scripts/mobile_ai_pick_place.py
```

### Target Following Demo

Real-time end-effector tracking using differential IK:

```bash
python3 trossen_arm_mujoco/scripts/wxai_follow_target.py
```

**Mouse Controls:**
- Double-click to select/highlight the target cube
- Ctrl + Left-click to rotate the view
- Ctrl + Right-click to move the target cube

The robot will track the target cube position in real-time.

---

## Controller API

The differential inverse kinematics controller (`IKController` in [`ik_controller.py`](./trossen_arm_mujoco/scripts/ik_controller.py)) provides Cartesian end-effector control for all Trossen AI robots.

### Key Features

- Damped least squares differential IK for smooth motion
- Gripper control with open/close commands
- Support for all robot types (WidowX AI, Stationary AI, Mobile AI)
- Position-only or full 6D pose control

### Basic Usage

```python
from trossen_arm_mujoco.scripts.ik_controller import IKController

# Initialize controller
robot = IKController(
    model=mujoco_model,
    data=mujoco_data,
    robot_type="wxai",  # or "stationary_ai", "mobile_ai"
    arm_joint_names=["joint_1", "joint_2", ...],
    gripper_joint_names=["left_carriage_joint"],
    ik_scale=1.0,
    ik_damping=0.03,
)

# Command end-effector pose
error = robot.set_ee_pose(
    target_position=np.array([0.3, 0.0, 0.2]),
    target_orientation=np.array([1.0, 0.0, 0.0, 0.0]),  # [w, x, y, z]
    position_only=False,
)

# Gripper control
robot.open_gripper()
robot.close_gripper()
```

### Controller Parameters

- `ik_scale`: Scaling factor for IK velocity (default: 1.0)
- `ik_damping`: Damping factor for singularity avoidance (default: 0.03)
- `position_only`: When True, only controls position, orientation remains free

---

## Data Collection

### Modules ([`trossen_arm_mujoco`](./trossen_arm_mujoco/))

This folder contains all Python modules necessary for running simulations, executing policies, recording episodes, and visualizing results.

### Simulations

- `ee_sim_env.py`
  - Loads `stationary_ai/scene_mocap.xml` (motion capture-based control).
  - The arms move by following the positions commanded to the mocap bodies.
  - Used for generating scripted policies that control the robot's arms in predefined ways.

- `sim_env.py`
  - Loads `stationary_ai/scene_joint.xml` (position-controlled joints).
  - Uses joint controllers instead of mocap bodies.
  - Replays joint trajectories from `ee_sim_env.py`, enabling clean simulation visuals without mocap bodies visible in the rendered output.

### Scripted Policy Execution

- `scripted_policy.py`
  - Defines pre-scripted movements for the robot arms to perform tasks like picking up objects.
  - Uses the motion capture bodies to generate smooth movement trajectories.
  - In the current setup, a policy is designed to pick up a red block, with randomized block positions in the environment.

## Data Collection Workflow

The data collection process involves two simulation phases:

1. Running a scripted policy in `ee_sim_env.py` to record observations (joint positions).
2. Replaying the recorded joint positions in `sim_env.py` to capture full episode data.

### Step-by-Step Process

1. Run `record_sim_episodes.py`

    - Starts `ee_sim_env.py` and executes a scripted policy.
    - Captures observations in the form of joint positions.
    - Saves these joint positions for later replay.
    - Immediately replays the episode in sim_env.py using the recorded joint positions.
    - During replay, captures:
      - Camera feeds from 4 different viewpoints
      - Joint states (actual positions during execution)
      - Actions (input joint positions)
      - Reward values indicating success or failure

2. Save the Data

    - All observations and actions are stored in HDF5 format, with one file per episode.
    - Each episode is saved as `episode_X.hdf5` inside the `~/.trossen/mujoco/data/` folder.

3. Visualizing the Data

    - The stored HDF5 files can be converted into videos using `visualize_eps.py`.

4. Sim-to-real

    - Run `replay_episode_real.py`
    - This script:
      - Loads the joint position trajectory from a selected HDF5 file.
      - Sends commands to both arms using IP addresses (--left_ip, --right_ip).
      - Plays back the motions based on the saved trajectory.
      - Monitors position error between commanded and actual joint states.
      - Returns arms to home and sleep positions after execution.


## Script Arguments

### a. record_sim_episodes.py

This script generates and saves demonstration episodes using a scripted policy in simulation. It supports both end-effector control (for task definition) and joint-space replay (for clean data collection), storing all observations in `.hdf5` format.

To generate and save simulation episodes, use:

```bash
python trossen_arm_mujoco/scripts/record_sim_episodes.py \
    --task_name sim_transfer_cube \
    --data_dir sim_transfer_cube \
    --num_episodes 5 \
    --onscreen_render
```
Arguments:

- `--task_name`: Name of the task (default: sim_transfer_cube).
- `--num_episodes`: Number of episodes to generate.
- `--data_dir`: Directory where episodes will be saved (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
- `--episode_len`: Number of simulation steps of each episode.
- `--onscreen_render` : Enables on-screen rendering. Default: False (only true if explicitly set)
- `--inject_noise`: Injects noise into actions. Default: False (only true if explicitly set)
- `--cam_names`: Comma-separated list of camera names for image collection

**Note:**

- When you pass `--task_name`, the script will automatically load the corresponding configuration from constants.py.

- You can extend `SIM_TASK_CONFIGS` in `constants.py` to support new task configurations.

- All parameters loaded from `constants.py` can be individually overridden via command-line arguments.

### b. visualize_eps.py

To convert saved episodes to videos, run:

```bash
python trossen_arm_mujoco/scripts/visualize_eps.py \
    --data_dir sim_transfer_cube \
    --output_dir videos \
    --fps 50
```
Arguments:

- `--data_dir`: Directory containing .hdf5 files (required), relative to --root_dir if provided.
- `--root_dir`: Root path prefix for locating data_dir. Default: ~/.trossen/mujoco/data/
- `--output_dir`: Subdirectory inside data_dir where generated .mp4 videos will be saved. Default: videos
- `--fps`: Frames per second for the generated videos (default: 50)
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/`

**Note:** If you do not specify `--root_dir`, videos will be saved to `~/.trossen/mujoco/data/<data_dir>/<output_dir>`.
You can customize the output path by changing `--root_dir`, `--data_dir`, or `--output_dir` as needed.

### c. replay_episode_real.py

This script replays recorded joint-space episodes on real Trossen robotic arms using data saved in .hdf5 files.
It configures each arm, plays back the actions with a user-defined frame rate, and returns both arms to a safe rest pose after execution.

To perform sim to real, run:

```bash
python trossen_arm_mujoco/scripts/replay_episode_real.py \
    --data_dir sim_transfer_cube \
    --episode_idx 0 \
    --fps 10 \
    --left_ip 192.168.1.5 \
    --right_ip 192.168.1.4
```

Arguments:

- `--data_dir`: Directory containing `.hdf5` files (required).
- `--root_dir`: Directory where the root is (optional). Default: `~/.trossen/mujoco/data/`
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
- `gripper`: The target gripper finger position 0~0.044 where 0 is closed and 0.044 is fully open.

Example:

```python
def generate_trajectory(self, ts_first: TimeStep):
    self.left_trajectory = [
        {"t": 0, "xyz": [0, 0, 0.4], "quat": [1, 0, 0, 0], "gripper": 0},
        {"t": 100, "xyz": [0.1, 0, 0.3], "quat": [1, 0, 0, 0], "gripper": 0.044}
    ]
```

### 3. Adding New Environment Setups

The simulation uses XML files stored in the `assets/` directory. To introduce a new environment setup:

1. Create a new XML configuration file in `assets/` with desired object placements and constraints.
2. Modify `sim_env.py` to load the new environment by specifying the new XML file.
3. Update the scripted policies in `scripted_policy.py` to accommodate new task goals and constraints.

---

## Troubleshooting

If you encounter into Mesa Loader or `mujoco.FatalError: gladLoadGL error` errors:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

---

## Related Links

- [Trossen Robotics](https://www.trossenrobotics.com/)
- [Trossen Arm Documentation](https://docs.trossenrobotics.com/trossen_arm/)
- [Trossen Arm Description (URDF)](https://github.com/TrossenRobotics/trossen_arm_description)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
