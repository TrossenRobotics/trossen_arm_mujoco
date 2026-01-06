# Stationary AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the Stationary AI bimanual setup by [Trossen Robotics](https://www.trossenrobotics.com/). It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and also uses the wxai_base.xml arm model.

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
5. Added equality constraints for gripper mimic joints (both arms).
6. Added position-controlled actuators with tuned PD gains (kp/kv) and force limits for all joints, plus armature and frictionloss for realistic motor dynamics.
7. Added keyframe for home position initialization.