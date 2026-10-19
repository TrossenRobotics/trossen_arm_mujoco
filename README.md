# Trossen Arm MuJoCo

## Overview

This repository provides MuJoCo simulation environments for Trossen AI robotic arms. It includes comprehensive robot models, inverse kinematics-based control, and demonstration scripts.

### What This Repository Offers

- MuJoCo XML models for Trossen AI robots:
  - WidowX AI (single arm manipulator)
  - Stationary AI (dual-arm stationary platform)
  - Mobile AI (dual-arm mobile manipulator)
- Differential inverse kinematics controller for Cartesian end-effector control
- Pick-and-place and target following demonstration scripts

## Index

- [Overview](#overview)
- [Robot Assets](#robot-assets)
- [Installation](#installation)
- [MuJoCo IK Demo Scripts](#mujoco-ik-demo-scripts)
- [Controller API](#controller-api)
- [Troubleshooting](#troubleshooting)
- [Related Links](#related-links)

## Robot Assets

All robot models are located in `assets/`:

```
assets/
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
uv run scripts/wxai_pick_place.py
```

If the simulation window appears with the robot performing pick-and-place, the setup was successful.

## MuJoCo IK Demo Scripts

These scripts demonstrate inverse kinematics-based control for manipulation tasks.

### Pick and Place Demonstrations

**WidowX AI** - Single arm pick-and-place:
```bash
uv run scripts/wxai_pick_place.py
```

**Stationary AI** - Dual-arm pick-and-place with handoff:
```bash
uv run scripts/stationary_ai_pick_place.py
```

**Mobile AI** - Mobile base navigation + dual-arm manipulation:
```bash
uv run scripts/mobile_ai_pick_place.py
```

### Target Following Demo

Real-time end-effector tracking using differential IK:

```bash
uv run scripts/wxai_follow_target.py
```

**Mouse Controls:**
- Double-click to select/highlight the target cube
- Ctrl + Left-click to rotate the view
- Ctrl + Right-click to move the target cube

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
