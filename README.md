# Trossen Arm MuJoCo

## Overview

This repository provides MuJoCo simulation environments for Trossen AI robotic arms. It includes comprehensive robot models, inverse kinematics-based control, and tools for simulation data collection and policy evaluation.

### What This Repository Offers

- MuJoCo XML models for Trossen AI robots:
  - WidowX AI (single arm manipulator)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Differential inverse kinematics controller for Cartesian end-effector control
- Pick-and-place and target following demonstration scripts
- Simulation data collection and policy evaluation for imitation learning

## Index

- [Overview](#overview)
- [Robot Assets](#robot-assets)
- [Installation](#installation)
- [MuJoCo IK Demo Scripts](#mujoco-ik-demo-scripts)
- [Controller API](#controller-api)
- [Simulation Data Collection and Evaluation](#simulation-data-collection-and-evaluation)
- [Troubleshooting](#troubleshooting)
- [Related Links](#related-links)

## Robot Assets

All robot models are located in `src/trossen_arm_mujoco/assets/`:

```
src/trossen_arm_mujoco/assets/
├── meshes/                                  # STL/OBJ files for robot components
├── wxai/
│   ├── wxai_base.xml                        # WidowX AI base model
│   ├── wxai_follower.xml                    # WidowX AI follower arm with camera
│   ├── scene.xml                            # Basic visualization scene
│   ├── scene_wxai_pick_place.xml            # Pick-and-place scene
│   └── scene_wxai_follow_target.xml         # Target following scene
├── stationary_ai/
│   ├── stationary_ai.xml                    # Dual-arm stationary platform
│   ├── scene.xml                            # Basic visualization scene
│   └── scene_stationary_ai_pick_place.xml   # Pick-and-place with handoff
└── mobile_ai/
    ├── mobile_ai.xml                        # Mobile base + dual-arm
    ├── scene.xml                            # Basic visualization scene
    └── scene_mobile_ai_pick_place.xml       # Mobile pick-and-place
```

All models are derived from URDF descriptions in [TrossenRobotics/trossen_arm_description](https://github.com/TrossenRobotics/trossen_arm_description). See individual `README.md` files in asset folders for detailed URDF→MJCF derivation steps.

## Installation

This project uses [uv](https://docs.astral.sh/uv/) to manage Python and dependencies.

Clone Repository

```bash
git clone https://github.com/TrossenRobotics/trossen_arm_mujoco.git
cd trossen_arm_mujoco
```

Install Dependencies

```bash
uv sync
```

Verify Installation

```bash
uv run wxai_pick_place
```

If the simulation window appears with the robot performing pick-and-place, the setup was successful.

## MuJoCo IK Demo Scripts

These scripts demonstrate inverse kinematics-based control for manipulation tasks.

### Pick and Place Demonstrations

**WidowX AI** - Single arm pick-and-place:
```bash
uv run wxai_pick_place
```

**Stationary AI** - Dual-arm pick-and-place with handoff:
```bash
uv run stationary_ai_pick_place
```

**Mobile AI** - Mobile base navigation + dual-arm manipulation:
```bash
uv run mobile_ai_pick_place
```

### Target Following Demo

Real-time end-effector tracking using differential IK:

```bash
uv run wxai_follow_target
```

**Mouse Controls:**
- Double-click the target cube to select it
- Ctrl + Right-click drag to move the cube
- Ctrl + Left-click drag to rotate the cube

The robot will track the target cube position in real-time.

## Controller API

The differential inverse kinematics controller (`Controller` in [`controller.py`](./src/trossen_arm_mujoco/controller.py)) provides Cartesian end-effector control for all Trossen AI robots.

### Key Features

- Damped least squares differential IK for smooth motion
- Gripper control with open/close commands
- Support for all robot types (WidowX AI, Stationary AI, Mobile AI)
- Position-only or full 6D pose control

### Basic Usage

```python
from trossen_arm_mujoco.controller import Controller, RobotType

# Initialize controller
robot = Controller(
    model=mujoco_model,
    data=mujoco_data,
    robot_type=RobotType.WXAI,  # or RobotType.STATIONARY_AI, RobotType.MOBILE_AI
    ee_site_name="ee_site",
    arm_joint_names=["joint_0", "joint_1", ...],
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

## Simulation Data Collection and Evaluation

Record scripted demonstrations and evaluate trained policies on the same simulated task, following the Simulation Transfer Cube setup of [ACT/ALOHA](https://arxiv.org/abs/2304.13705) (Sec. V). Episodes use the ACT/ALOHA HDF5 format.

**Task (Stationary AI transfer cube):** the left arm picks a cube from a random position, hands it over to the right arm, and the right arm places it on the target.

The shared environment ([`transfer_cube_env.py`](./src/trossen_arm_mujoco/transfer_cube_env.py)) runs at 50 Hz:

- **Action:** 14 absolute joint targets, 7 per arm (6 joints + gripper), left arm first
- **Observation:** `qpos` and `qvel` of the same 14 joints, plus 480x640 images from `cam_high`, `cam_low`, `follower_left_camera`, `follower_right_camera`
- **Reward:** 1 touch, 2 lift, 3 handover, 4 place (success)

### 1. Record Demonstrations

Runs the Stationary AI pick-and-place demo as a scripted policy and saves only successful episodes:

```bash
uv run stationary_ai_transfer_cube_record --num_episodes 50
```

Episodes are saved to `data/stationary_ai_transfer_cube/episode_N.hdf5` in the current directory:

```
/action                          (T, 14)
/observations/qpos               (T, 14)
/observations/qvel               (T, 14)
/observations/images/<cam_name>  (T, 480, 640, 3)
```

Options: `--root_dir`, `--data_dir`, `--episode_len` (default 1200), `--seed`, `--inject_noise`, `--onscreen_render`, `--cam_names`.

### 2. Convert an Episode to Video

```bash
uv run episode_to_video data/stationary_ai_transfer_cube/episode_0.hdf5
```

The video, with all cameras side by side, is saved next to the episode as `episode_0.mp4`.

### 3. Evaluate a Policy

Runs a policy on random cube positions and reports the success rate of each subtask:

```bash
uv run stationary_ai_transfer_cube_eval --policy scripted --num_trials 50
```

To evaluate your own policy (e.g. a trained ACT model), add it to `make_policy()` in the script. A policy needs `reset()` and `__call__(observation)` returning the 14 joint targets.

Options: `--num_trials`, `--episode_len`, `--seed`, `--onscreen_render`, `--cam_names` (pass `--cam_names` with no names to skip rendering).

## Troubleshooting

### Slow Recording

Recording should take about 30 s per episode. If it is 10-50x slower, MuJoCo is rendering on the CPU. Check with:

```bash
uv run python3 -c 'import mujoco; from OpenGL import GL; r = mujoco.Renderer(mujoco.MjModel.from_xml_string("<mujoco/>")); print(GL.glGetString(GL.GL_RENDERER).decode())'
```

If it prints `llvmpipe` on a laptop with an NVIDIA GPU, render on the NVIDIA GPU:

```bash
export __NV_PRIME_RENDER_OFFLOAD=1
export __GLX_VENDOR_LIBRARY_NAME=nvidia
```

On a headless server, use `export MUJOCO_GL=egl`.

## Related Links

- [Trossen Robotics](https://www.trossenrobotics.com/)
- [Trossen Arm Documentation](https://docs.trossenrobotics.com/trossen_arm/)
- [Trossen Arm Description (URDF)](https://github.com/TrossenRobotics/trossen_arm_description)
- [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- [MuJoCo Python Bindings](https://mujoco.readthedocs.io/en/stable/python.html)
