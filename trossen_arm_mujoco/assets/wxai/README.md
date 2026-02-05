# WXAI Base and WXAI Follower Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the [Trossen Robotics WidowX AI](https://www.trossenrobotics.com/widowx-ai) arms. It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description).

- **wxai_base.xml** - Base WXAI arm without camera mount
- **wxai_follower.xml** - WXAI arm with D405 camera mount and camera

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Added base_link as a proper body element with inertial properties.
3. Added default classes for visual and collision geoms.
4. Added simplified collision geometries using primitive shapes.
5. Added camera_color_frame site and camera matching the URDF's camera frame position. MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Camera is rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF. Camera fovy values are adjusted as they are not specified in URDF.
6. Added end-effector site (`ee_site`) at gripper tip for pose control and trajectory planning.
7. Added gravity compensation (`gravcomp="1"` on bodies, `actuatorgravcomp="true"` on joint actuators) to counteract gravitational forces and improve control stability, mimicking real hardware behavior.
8. Added equality constraint for gripper mimic joint.
9. Added position-controlled actuators with PD gains (kp/kv) for all 6 joints + gripper, plus armature and frictionloss for motor dynamics. **Note:** Actuator parameters (PD gains, armature, frictionloss) are tuned for simulation. MuJoCo's actuator model differs from real hardware due to factors like gravity compensation, solver timestep rates, and control loop differences, making manufacturer specifications not directly applicable.
10. Added keyframe for home position initialization.
