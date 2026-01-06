# Mobile AI Bimanual Description (MJCF)

## Overview

This package contains robot descriptions (MJCF) of the Mobile AI bimanual setup by [Trossen Robotics](https://www.trossenrobotics.com/). It is derived from the [URDF description](https://github.com/TrossenRobotics/trossen_arm_description) and also uses the wxai_base.xml arm model.

- **mobile_ai.xml** - Mobile base with bimanual WXAI arm setup

## URDF → MJCF Derivation Steps

1. Converted URDF to MuJoCo XML.
2. Followed wxai_base.xml structure for left and right arms (follower_left, follower_right) with appropriate positions and orientations.
3. Added simplified collision geometries for mobile base and wheels using primitive shapes.
4. Added three cameras:
   - `cam_high` - External overhead camera mounted on frame
   - `cam_left_wrist` - Wrist-mounted camera on left arm
   - `cam_right_wrist` - Wrist-mounted camera on right arm
5. Added equality constraints for gripper mimic joints (both arms).
6. Added PD gains and force limits for arm actuators, and velocity-controlled wheel actuators.
7. Added keyframe for home position initialization.
