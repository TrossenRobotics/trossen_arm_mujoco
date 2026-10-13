# Stationary AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the [Trossen Robotics Stationary AI](https://www.trossenrobotics.com/stationary-ai) bimanual setup. It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and attaches the `wxai_follower.xml` arm model.

- **stationary_ai.xml** - Bimanual WXAI arm setup

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Attached `wxai/wxai_follower.xml` for the left and right arms with `<attach>` (prefixes `follower_left_`, `follower_right_`) at the URDF mount poses. Arm sites, gravity compensation, mimic constraint and actuators come from that file.
3. Added simplified collision geometries for frame_link and tabletop_link using primitive shapes.
4. Added four cameras:
   - `cam_high` - External overhead camera mounted on frame
   - `cam_low` - External low-angle camera mounted on frame
   - `follower_left_camera` - Wrist-mounted camera on left arm
   - `follower_right_camera` - Wrist-mounted camera on right arm
   - **Note:** MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Cameras are rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF. Camera fovy values are adjusted as they are not specified in URDF.
5. The `tabletop_link` inertial combines the URDF tabletop, frame, camera mounts and D405 cameras.
