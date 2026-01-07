# Stationary AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the [Trossen Robotics Stationary AI](https://www.trossenrobotics.com/stationary-ai) bimanual setup. It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and also uses the wxai_base.xml arm model.

- **stationary_ai.xml** - Bimanual WXAI arm setup

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Followed wxai_base.xml structure for left and right arms (follower_left, follower_right) with appropriate positions and orientations.
3. Added simplified collision geometries for frame_link and tabletop_link using primitive shapes.
4. Added four cameras:
   - `cam_high` - External overhead camera mounted on frame
   - `cam_low` - External low-angle camera mounted on frame
   - `cam_left_wrist` - Wrist-mounted camera on left arm
   - `cam_right_wrist` - Wrist-mounted camera on right arm
   - **Note:** MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Cameras are rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF. Camera fovy values are adjusted as they are not specified in URDF.
5. Added equality constraints for gripper mimic joints (both arms).
6. Added position-controlled actuators with tuned PD gains (kp/kv) and force limits for all joints, plus armature and frictionloss for realistic motor dynamics. **Note:** Actuator parameters (PD gains, armature, frictionloss) are tuned for simulation. MuJoCo's actuator model differs from real hardware due to factors like gravity compensation, solver timestep rates, and control loop differences, making manufacturer specifications not directly applicable.
7. Added keyframe for home position initialization.