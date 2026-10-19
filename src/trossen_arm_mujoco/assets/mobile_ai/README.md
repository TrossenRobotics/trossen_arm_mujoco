# Mobile AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the [Trossen Robotics Mobile AI](https://www.trossenrobotics.com/mobile-ai) bimanual setup. It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and attaches the `wxai_follower.xml` arm model.

- **mobile_ai.xml** - Mobile base with bimanual WXAI arm setup

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Attached `wxai/wxai_follower.xml` for the left and right arms with `<attach>` (prefixes `follower_left_`, `follower_right_`) at the URDF mount poses. Arm sites, gravity compensation, mimic constraint and actuators come from that file.
3. Added simplified collision geometries for mobile base and wheels using primitive shapes.
4. Added three cameras:
   - `cam_high` - External overhead camera mounted on frame
   - `follower_left_camera` - Wrist-mounted camera on left arm
   - `follower_right_camera` - Wrist-mounted camera on right arm
   - **Note:** MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Cameras are rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF. Camera fovy values are adjusted as they are not specified in URDF.
5. The `mobile_base_link` body frame is the URDF `base_footprint` (on the ground), so the base inertial is expressed in that frame.
6. Split the URDF `base_link.dae` by material into `base_link_color_{0,1,2}.obj` (black, white, steel) to keep its colors.
7. Added velocity-controlled wheel actuators.
