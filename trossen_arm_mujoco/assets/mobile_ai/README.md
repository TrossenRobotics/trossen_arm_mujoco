# Mobile AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the [Trossen Robotics Mobile AI](https://www.trossenrobotics.com/mobile-ai) bimanual setup. It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and also uses the wxai_base.xml arm model.

- **mobile_ai.xml** - Mobile base with bimanual WXAI arm setup

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Followed wxai_base.xml structure for left and right arms (follower_left, follower_right) with appropriate positions and orientations.
3. Added simplified collision geometries for mobile base and wheels using primitive shapes.
4. Added three cameras:
   - `cam_high` - External overhead camera mounted on frame
   - `cam_left_wrist` - Wrist-mounted camera on left arm
   - `cam_right_wrist` - Wrist-mounted camera on right arm
   - **Note:** MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Cameras are rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF. Camera fovy values are adjusted as they are not specified in URDF.
5. Added end-effector sites (`follower_left_ee_site`, `follower_right_ee_site`) at gripper tips for pose control and trajectory planning.
6. Added gravity compensation (`gravcomp="1"` on arm bodies, `actuatorgravcomp="true"` on arm joint actuators) to counteract gravitational forces and improve control stability, mimicking real hardware behavior.
7. Added equality constraints for gripper mimic joints (both arms).
8. Added PD gains and force limits for arm actuators, and velocity-controlled wheel actuators. **Note:** Actuator parameters (PD gains, armature, frictionloss) are tuned for simulation. MuJoCo's actuator model differs from real hardware due to factors like gravity compensation, solver timestep rates, and control loop differences, making manufacturer specifications not directly applicable.
9. Added keyframe for home position initialization.

## TODO

- Add texture files for mobile_base_link visual mesh.
