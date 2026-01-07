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
   - **Note:** MuJoCo cameras are oriented along the +Z axis, while URDF camera frames point along the +X axis. Cameras are rotated +90° around Y, then +90° around Z (in camera frame) to align with URDF.
5. Added equality constraints for gripper mimic joints (both arms).
6. Added PD gains and force limits for arm actuators, and velocity-controlled wheel actuators.
7. Added keyframe for home position initialization.

## TODO

- Add texture files for mobile_base_link visual mesh.
