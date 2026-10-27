# Copyright 2026 Trossen Robotics
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""Trossen Base Controller.

Provides velocity control for the Rivet swerve drive base, converting a body
frame velocity command (vx, vy, omega) into pivot angles and wheel speeds, and
position control for the lift.
"""

from typing import Tuple

import mujoco
import numpy as np

# Swerve module names, each with a "<name>_pivot" and "<name>_drive" joint and actuator
DEFAULT_MODULE_NAMES = ["b", "fr", "fl"]

# Wheel radius in meters
DEFAULT_WHEEL_RADIUS = 0.0535

# Default velocity and acceleration limits
DEFAULT_VEL_LINEAR_MAX = 1.0  # m/s
DEFAULT_VEL_ANGULAR_MAX = 3.0  # rad/s
DEFAULT_ACCEL_LINEAR_MAX = 2.5  # m/s^2
DEFAULT_ACCEL_ANGULAR_MAX = 9.0  # rad/s^2
DEFAULT_VEL_DEADZONE = 0.05  # m/s

# Default lift limits
DEFAULT_LIFT_VEL_MAX = 0.2  # m/s
DEFAULT_LIFT_ACCEL_MAX = 0.5  # m/s^2
DEFAULT_LIFT_DECEL_MAX = 1.0  # m/s^2


class BaseController:
    """Velocity controller for a swerve drive base.

    Every update:

    - Limit the target speed to what the base is allowed to do
    - Ramp toward the target so the base speeds up and slows down smoothly (robot acc limit)
    - Work out how fast and in which direction each wheel has to move
    - Slow all wheels down together if any wheel would go faster than it can
    - Turn each pivot the shortest way, running the wheel backward when that is quicker
    - Stop a wheel and hold its pivot when it barely needs to move
    - Move the lift toward its target height, speeding up and slowing down smoothly
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        base_body_name: str = "base_link",
        module_names: list = None,
        wheel_radius: float = DEFAULT_WHEEL_RADIUS,
        vel_linear_max: float = DEFAULT_VEL_LINEAR_MAX,
        vel_angular_max: float = DEFAULT_VEL_ANGULAR_MAX,
        accel_linear_max: float = DEFAULT_ACCEL_LINEAR_MAX,
        accel_angular_max: float = DEFAULT_ACCEL_ANGULAR_MAX,
        vel_deadzone: float = DEFAULT_VEL_DEADZONE,
        lift_name: str = "lift",
        lift_vel_max: float = DEFAULT_LIFT_VEL_MAX,
        lift_accel_max: float = DEFAULT_LIFT_ACCEL_MAX,
        lift_decel_max: float = DEFAULT_LIFT_DECEL_MAX,
    ):
        """Initialize base controller.

        :param model: MuJoCo model
        :param data: MuJoCo data
        :param base_body_name: Name of the base body the modules are attached to
        :param module_names: List of swerve module names. If None, uses defaults.
        :param wheel_radius: Wheel radius in meters
        :param vel_linear_max: Max commanded linear speed (m/s)
        :param vel_angular_max: Max commanded angular speed (rad/s)
        :param accel_linear_max: Max linear acceleration (m/s^2)
        :param accel_angular_max: Max angular acceleration (rad/s^2)
        :param vel_deadzone: Module speed below which the wheel stops and the pivot holds (m/s)
        :param lift_name: Name of the lift joint and its position actuator
        :param lift_vel_max: Max lift speed (m/s)
        :param lift_accel_max: Max lift acceleration when speeding up (m/s^2)
        :param lift_decel_max: Max lift deceleration when slowing down (m/s^2)
        """
        self.model = model
        self.data = data

        self.wheel_radius = wheel_radius
        self.vel_linear_max = vel_linear_max
        self.vel_angular_max = vel_angular_max
        self.accel_linear_max = accel_linear_max
        self.accel_angular_max = accel_angular_max
        self.vel_deadzone = vel_deadzone
        self.lift_vel_max = lift_vel_max
        self.lift_accel_max = lift_accel_max
        self.lift_decel_max = lift_decel_max

        if module_names is None:
            module_names = DEFAULT_MODULE_NAMES

        base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, base_body_name)
        if base_body_id < 0:
            raise ValueError(f"Base body '{base_body_name}' not found")
        self.base_body_id = base_body_id

        # Get module positions, joint addresses and actuator IDs
        module_positions = []
        self.pivot_qpos_indices = []
        self.pivot_actuator_ids = []
        self.drive_actuator_ids = []
        self.wheel_velocity_max = []

        for name in module_names:
            pivot_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_pivot")
            if pivot_joint_id < 0:
                raise ValueError(f"Pivot joint '{name}_pivot' not found")

            # Module (x, y) relative to the base, from the pivot body position
            pivot_body_id = model.jnt_bodyid[pivot_joint_id]
            if model.body_parentid[pivot_body_id] != base_body_id:
                raise ValueError(f"Pivot body for '{name}' is not a child of '{base_body_name}'")
            module_positions.append(model.body_pos[pivot_body_id][:2].copy())

            self.pivot_qpos_indices.append(model.jnt_qposadr[pivot_joint_id])

            pivot_actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_pivot"
            )
            drive_actuator_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_drive"
            )
            if pivot_actuator_id < 0 or drive_actuator_id < 0:
                raise ValueError(f"Actuators '{name}_pivot' and '{name}_drive' are required")
            self.pivot_actuator_ids.append(pivot_actuator_id)
            self.drive_actuator_ids.append(drive_actuator_id)

            self.wheel_velocity_max.append(model.actuator_ctrlrange[drive_actuator_id][1])

        self.module_positions = np.array(module_positions)
        self.num_modules = len(module_names)

        self.module_speed_max = min(self.wheel_velocity_max) * self.wheel_radius

        # Get lift joint and actuator
        lift_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, lift_name)
        self.lift_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, lift_name)
        if lift_joint_id < 0 or self.lift_actuator_id < 0:
            raise ValueError(f"Lift joint and actuator '{lift_name}' are required")
        self.lift_qpos_index = model.jnt_qposadr[lift_joint_id]
        self.lift_range = model.actuator_ctrlrange[self.lift_actuator_id].copy()

        self.reset()

    def reset(self) -> None:
        """Stop the base, hold the pivots at their current angles and hold the lift where it is."""
        # Target (vx, vy, omega) and the accel-limited command actually sent
        self.target_velocity = np.zeros(3)
        self.command_velocity = np.zeros(3)

        for i in range(self.num_modules):
            self.data.ctrl[self.pivot_actuator_ids[i]] = self.data.qpos[self.pivot_qpos_indices[i]]
            self.data.ctrl[self.drive_actuator_ids[i]] = 0.0

        # Lift setpoint sent to the actuator
        self.lift_setpoint = self.get_lift_position()
        self.lift_command_velocity = 0.0
        self.lift_goal = self.lift_setpoint
        self.lift_cruise_velocity = self.lift_vel_max
        self.data.ctrl[self.lift_actuator_id] = self.lift_setpoint

    def set_velocity(self, vx: float, vy: float, omega: float) -> None:
        """Set the target base velocity in the base frame.

        The linear velocity is scaled to at most vel_linear_max, keeping its
        direction, and the angular velocity is clipped to vel_angular_max.

        :param vx: Forward velocity (m/s)
        :param vy: Left velocity (m/s)
        :param omega: Counterclockwise angular velocity (rad/s)
        """
        speed = np.hypot(vx, vy)
        if speed > self.vel_linear_max:
            vx, vy = vx * self.vel_linear_max / speed, vy * self.vel_linear_max / speed
        omega = np.clip(omega, -self.vel_angular_max, self.vel_angular_max)

        self.target_velocity = np.array([vx, vy, omega])

    def stop(self) -> None:
        """Ramp the base down to a stop."""
        self.set_velocity(0.0, 0.0, 0.0)

    def get_command_velocity(self) -> np.ndarray:
        """Get the accel-limited velocity currently being commanded.

        :return: Commanded velocity [vx, vy, omega]
        """
        return self.command_velocity.copy()

    def get_velocity(self) -> np.ndarray:
        """Get the actual base velocity in the base frame.

        :return: Base velocity [vx, vy, omega]
        """
        # Body velocity in its own frame
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, self.base_body_id, velocity, 1
        )
        return np.array([velocity[3], velocity[4], velocity[2]])

    def set_lift_position(self, position: float, velocity: float = None) -> None:
        """Move the lift to a height, cruising at a speed.

        :param position: Target lift height (m), clipped to the lift range
        :param velocity: Cruise speed (m/s). If None, uses lift_vel_max.
        """
        if velocity is None:
            velocity = self.lift_vel_max
        self.lift_goal = float(np.clip(position, self.lift_range[0], self.lift_range[1]))
        self.lift_cruise_velocity = min(abs(velocity), self.lift_vel_max)

    def get_lift_position(self) -> float:
        """Get the actual lift height.

        :return: Lift height (m)
        """
        return float(self.data.qpos[self.lift_qpos_index])

    def get_pivot_angles(self) -> np.ndarray:
        """Get current pivot angles.

        :return: Pivot angles (rad)
        """
        return np.array([self.data.qpos[idx] for idx in self.pivot_qpos_indices])

    def update(self, dt: float = None) -> None:
        """Compute and apply one control step.

        Should be called every control step, before stepping the simulation.

        :param dt: Time since the last update (s). If None, uses the model timestep.
        """
        if dt is None:
            dt = self.model.opt.timestep

        self.limit_acceleration(dt)
        vx, vy, omega = self.command_velocity

        # Velocity of each module: v + omega x r
        module_velocities = np.column_stack(
            [
                vx - self.module_positions[:, 1] * omega,
                vy + self.module_positions[:, 0] * omega,
            ]
        )

        # If any wheel would go too fast, scale all wheels by the same factor
        # so they still agree on the motion and only the speed drops
        max_speed = np.max(np.linalg.norm(module_velocities, axis=1))
        if max_speed > self.module_speed_max:
            module_velocities *= self.module_speed_max / max_speed

        for i in range(self.num_modules):
            pivot_actuator_id = self.pivot_actuator_ids[i]
            angle, wheel_velocity = self.compute_module_command(
                module_velocities[i], self.data.ctrl[pivot_actuator_id]
            )
            self.data.ctrl[pivot_actuator_id] = angle
            self.data.ctrl[self.drive_actuator_ids[i]] = wheel_velocity

        self.update_lift(dt)

    def update_lift(self, dt: float) -> None:
        """Move the lift setpoint toward the goal height.

        - Cruise at the set speed, and slow down early enough to stop right at the goal
        - Speeding up is limited by lift_accel_max and slowing down by lift_decel_max

        :param dt: Time step (s)
        """
        error = self.lift_goal - self.lift_setpoint
        # Fastest speed that can still stop in the remaining distance
        speed = min(self.lift_cruise_velocity, np.sqrt(2 * self.lift_decel_max * abs(error)))
        desired_velocity = np.sign(error) * speed

        # Ramp the speed, speeding up gently and slowing down faster
        speeding_up = abs(desired_velocity) > abs(self.lift_command_velocity)
        max_step = (self.lift_accel_max if speeding_up else self.lift_decel_max) * dt
        self.lift_command_velocity += np.clip(
            desired_velocity - self.lift_command_velocity, -max_step, max_step
        )

        setpoint = self.lift_setpoint + self.lift_command_velocity * dt

        # Stop exactly at the goal instead of stepping past it
        if (setpoint - self.lift_goal) * np.sign(error) >= 0:
            setpoint = self.lift_goal
            self.lift_command_velocity = 0.0

        self.lift_setpoint = setpoint
        self.data.ctrl[self.lift_actuator_id] = self.lift_setpoint

    def limit_acceleration(self, dt: float) -> None:
        """Move the commanded velocity toward the target by at most accel * dt.

        vx and vy ramp together so the base keeps moving in a straight line.

        :param dt: Time step (s)
        """
        delta = self.target_velocity - self.command_velocity

        max_step = self.accel_linear_max * dt
        step = np.hypot(delta[0], delta[1])
        if step > max_step:
            delta[:2] *= max_step / step

        max_step = self.accel_angular_max * dt
        delta[2] = np.clip(delta[2], -max_step, max_step)

        self.command_velocity += delta

    def compute_module_command(
        self, module_velocity: np.ndarray, prev_angle: float
    ) -> Tuple[float, float]:
        """Convert a module velocity into a pivot angle and wheel speed.

        - Too slow: stop the wheel and keep the pivot where it is, since a tiny
          velocity has no meaningful direction and would make the pivot twitch
        - More than 90 deg away: point the pivot the opposite way and spin the
          wheel backward, which gives the same motion with a smaller turn
        - Always turn the shortest way from the previous angle, so the pivot
          never spins almost a full circle when crossing +-180 deg

        :param module_velocity: Module velocity [vx, vy] in the base frame (m/s)
        :param prev_angle: Previously commanded pivot angle (rad)
        :return: Tuple of (pivot angle in rad, wheel velocity in rad/s)
        """
        vx, vy = module_velocity

        # Too slow to have a meaningful direction, stop the wheel and hold the pivot angle
        if abs(vx) < self.vel_deadzone and abs(vy) < self.vel_deadzone:
            return prev_angle, 0.0

        angle = np.arctan2(vy, vx)
        wheel_velocity = np.hypot(vx, vy) / self.wheel_radius

        # Shortest signed turn from the previous angle, wrapped to [-pi, pi]
        diff = np.arctan2(np.sin(angle - prev_angle), np.cos(angle - prev_angle))

        # More than 90 deg away, turn to the opposite angle and spin the wheel backward instead
        if abs(diff) > np.pi / 2:
            diff = np.arctan2(np.sin(diff + np.pi), np.cos(diff + np.pi))
            wheel_velocity = -wheel_velocity

        # Aim relative to the previous angle so the pivot never takes the long way around
        return prev_angle + diff, wheel_velocity
