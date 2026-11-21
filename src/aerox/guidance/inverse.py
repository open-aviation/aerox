"""Inverse flight-mechanics control above the realized-attitude plant.

This module deliberately commands actual pitch, bank, and total realized net
thrust.  It is a model-based trajectory-generation layer, not an Airbus flight
control law, surface allocator, FADEC model, or identified autopilot.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike

from aerox.guidance.path import SpeedGuidanceMode
from aerox.models.atmosphere import G0, atmosphere_at_pressure_altitude
from aerox.modeling.dynamics import (
    PitchAwareLiftModel,
    PitchBankThrustInput,
    PointMassForces,
    PointMassModel,
    PointMassState,
)


def _wrapped_angle_error(target: float, actual: float) -> float:
    return float(np.arctan2(np.sin(target - actual), np.cos(target - actual)))


def _bounded_rate(
    target: float, previous: float | None, maximum_rate: float, step_s: float
) -> tuple[float, bool]:
    if previous is None:
        return target, False
    bound = maximum_rate * step_s
    value = float(np.clip(target, previous - bound, previous + bound))
    return value, not np.isclose(value, target, rtol=0.0, atol=1e-12)


@dataclass(frozen=True)
class PitchBankThrustCommand:
    """One generated command plus inversion and saturation diagnostics."""

    control: PitchBankThrustInput
    effective_reference_flight_path_angle_rad: float
    target_lift_coefficient: float
    achieved_lift_coefficient: float
    energy_limited: bool
    pitch_limited: bool
    bank_limited: bool
    thrust_limited: bool


@dataclass(frozen=True)
class InverseFlightMechanicsController:
    """Inverse-flight-mechanics feedforward with bounded state feedback.

    Gains and limits are supplied explicitly by the caller.  The controller
    contains no A320 defaults and makes no claim about proprietary Airbus laws.
    """

    speed_error_gain_per_s: float
    flight_path_angle_error_gain_per_s: float
    heading_error_gain_per_s: float
    altitude_to_flight_path_gain_rad_per_m: float
    minimum_pitch_rad: float
    maximum_pitch_rad: float
    maximum_bank_rad: float
    minimum_thrust_n: float
    maximum_thrust_n: float
    maximum_pitch_rate_rad_s: float
    maximum_bank_rate_rad_s: float
    maximum_thrust_rate_n_s: float

    def __post_init__(self) -> None:
        values = tuple(float(value) for value in self.__dict__.values())
        if any(not np.isfinite(value) for value in values):
            raise ValueError("controller gains and limits must be finite")
        if (
            min(
                self.speed_error_gain_per_s,
                self.flight_path_angle_error_gain_per_s,
                self.heading_error_gain_per_s,
                self.altitude_to_flight_path_gain_rad_per_m,
                self.maximum_pitch_rate_rad_s,
                self.maximum_bank_rate_rad_s,
                self.maximum_thrust_rate_n_s,
            )
            <= 0.0
        ):
            raise ValueError("controller gains and rate limits must be positive")
        if self.minimum_pitch_rad >= self.maximum_pitch_rad:
            raise ValueError("minimum pitch must be below maximum pitch")
        if not 0.0 < self.maximum_bank_rad < np.deg2rad(85.0):
            raise ValueError("maximum bank must be between zero and 85 degrees")
        if (
            self.minimum_thrust_n < 0.0
            or self.minimum_thrust_n >= self.maximum_thrust_n
        ):
            raise ValueError("thrust bounds are invalid")

    @staticmethod
    def _lift_coefficient_for_pitch(
        plant: PointMassModel,
        *,
        pitch_angle_rad: float,
        flight_path_angle_rad: float,
        mach: float,
    ) -> float:
        if isinstance(plant.lift_map, PitchAwareLiftModel):
            _, coefficient = plant.lift_map.angle_and_lift_coefficient_scalar(
                pitch_angle_rad=pitch_angle_rad,
                flight_path_angle_rad=flight_path_angle_rad,
                mach=mach,
            )
            return float(coefficient)
        return plant.lift_map.lift_coefficient_scalar(
            pitch_angle_rad - flight_path_angle_rad, mach
        )

    def _pitch_for_lift_coefficient(
        self,
        plant: PointMassModel,
        *,
        target_lift_coefficient: float,
        flight_path_angle_rad: float,
        mach: float,
    ) -> tuple[float, bool]:
        lower = self.minimum_pitch_rad
        upper = self.maximum_pitch_rad
        lower_cl = self._lift_coefficient_for_pitch(
            plant,
            pitch_angle_rad=lower,
            flight_path_angle_rad=flight_path_angle_rad,
            mach=mach,
        )
        upper_cl = self._lift_coefficient_for_pitch(
            plant,
            pitch_angle_rad=upper,
            flight_path_angle_rad=flight_path_angle_rad,
            mach=mach,
        )
        minimum_cl = min(lower_cl, upper_cl)
        maximum_cl = max(lower_cl, upper_cl)
        if target_lift_coefficient <= minimum_cl:
            return (lower if lower_cl <= upper_cl else upper), True
        if target_lift_coefficient >= maximum_cl:
            return (upper if upper_cl >= lower_cl else lower), True
        increasing = upper_cl > lower_cl
        for _ in range(48):
            middle = 0.5 * (lower + upper)
            middle_cl = self._lift_coefficient_for_pitch(
                plant,
                pitch_angle_rad=middle,
                flight_path_angle_rad=flight_path_angle_rad,
                mach=mach,
            )
            if (middle_cl < target_lift_coefficient) == increasing:
                lower = middle
            else:
                upper = middle
        return 0.5 * (lower + upper), False

    def command(
        self,
        *,
        plant: PointMassModel,
        state: PointMassState | ArrayLike,
        reference_true_airspeed_mps: float,
        reference_true_airspeed_rate_mps2: float,
        reference_flight_path_angle_rad: float,
        reference_flight_path_angle_rate_rad_s: float,
        reference_heading_rad: float,
        reference_heading_rate_rad_s: float,
        reference_pressure_altitude_m: float,
        altitude_error_feedback_gain_scale: float = 1.0,
        static_temperature_k: float,
        speed_guidance_mode: SpeedGuidanceMode = "PROFILE",
        minimum_available_thrust_n: float | None = None,
        maximum_available_thrust_n: float | None = None,
        operational_thrust_target_n: float | None = None,
        drag_coefficient_increment: float = 0.0,
        maximum_speed_acceleration_mps2: float | None = None,
        maximum_speed_deceleration_mps2: float | None = None,
        previous_control: PitchBankThrustInput | None = None,
        command_step_s: float = 1.0,
        wind_north_mps: float = 0.0,
        wind_east_mps: float = 0.0,
        wind_down_mps: float = 0.0,
    ) -> PitchBankThrustCommand:
        """Generate one bounded realized-attitude/net-thrust command."""

        values = plant.validate_state(state)
        if not np.isfinite(static_temperature_k) or static_temperature_k <= 0.0:
            raise ValueError("static_temperature_k must be finite and positive")
        if (
            not np.isfinite(altitude_error_feedback_gain_scale)
            or not 0.0 <= altitude_error_feedback_gain_scale <= 1.0
        ):
            raise ValueError(
                "altitude feedback gain scale must lie between zero and one"
            )
        if not np.isfinite(command_step_s) or command_step_s <= 0.0:
            raise ValueError("command_step_s must be finite and positive")
        if (
            not np.isfinite(drag_coefficient_increment)
            or drag_coefficient_increment < 0.0
        ):
            raise ValueError(
                "drag_coefficient_increment must be finite and non-negative"
            )
        if speed_guidance_mode not in (
            "PROFILE",
            "CAS_MACH",
            "MACH_CAS",
            "SPEED_HOLD",
        ):
            raise ValueError("speed_guidance_mode is invalid")
        minimum_thrust = self.minimum_thrust_n
        if minimum_available_thrust_n is not None:
            if (
                not np.isfinite(minimum_available_thrust_n)
                or minimum_available_thrust_n < 0.0
            ):
                raise ValueError(
                    "minimum available thrust must be finite and non-negative"
                )
            minimum_thrust = max(minimum_thrust, minimum_available_thrust_n)
        maximum_thrust = self.maximum_thrust_n
        if maximum_available_thrust_n is not None:
            if (
                not np.isfinite(maximum_available_thrust_n)
                or maximum_available_thrust_n <= 0.0
            ):
                raise ValueError("maximum available thrust must be finite and positive")
            maximum_thrust = min(maximum_thrust, maximum_available_thrust_n)
        if minimum_thrust > maximum_thrust:
            raise ValueError(
                "available maximum thrust must not be below minimum thrust"
            )
        if operational_thrust_target_n is not None and (
            not np.isfinite(operational_thrust_target_n)
            or not minimum_thrust <= operational_thrust_target_n <= maximum_thrust
        ):
            raise ValueError(
                "operational thrust target must lie in the available thrust range"
            )
        for label, limit in (
            ("maximum speed acceleration", maximum_speed_acceleration_mps2),
            ("maximum speed deceleration", maximum_speed_deceleration_mps2),
        ):
            if limit is not None and (not np.isfinite(limit) or limit <= 0.0):
                raise ValueError(f"{label} must be finite and positive")
        speed = values.true_airspeed_mps
        gamma = values.flight_path_angle_rad
        heading = values.heading_rad
        altitude = values.pressure_altitude_m
        mass = values.mass_kg
        requested_speed_acceleration = reference_true_airspeed_rate_mps2 + (
            self.speed_error_gain_per_s * (reference_true_airspeed_mps - speed)
        )
        speed_acceleration = float(
            np.clip(
                requested_speed_acceleration,
                (
                    -np.inf
                    if maximum_speed_deceleration_mps2 is None
                    else -maximum_speed_deceleration_mps2
                ),
                (
                    np.inf
                    if maximum_speed_acceleration_mps2 is None
                    else maximum_speed_acceleration_mps2
                ),
            )
        )
        speed_energy_limited = not np.isclose(
            speed_acceleration,
            requested_speed_acceleration,
            rtol=0.0,
            atol=1e-12,
        )
        altitude_gamma_correction = (
            altitude_error_feedback_gain_scale
            * self.altitude_to_flight_path_gain_rad_per_m
            * (reference_pressure_altitude_m - altitude)
        )
        heading_rate = reference_heading_rate_rad_s + self.heading_error_gain_per_s * (
            _wrapped_angle_error(reference_heading_rad, heading)
        )
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([altitude]), np.array([static_temperature_k])
        )
        mach = speed / float(atmosphere.speed_of_sound_mps[0])
        dynamic_pressure = 0.5 * float(atmosphere.density_kg_m3[0]) * speed**2

        def attitude_for_gamma_target(
            gamma_target_rad: float,
        ) -> tuple[float, float, float, PointMassForces, bool, bool]:
            gamma_rate = reference_flight_path_angle_rate_rad_s + (
                self.flight_path_angle_error_gain_per_s * (gamma_target_rad - gamma)
            )
            vertical_specific_force = speed * gamma_rate + G0 * np.cos(gamma)
            lateral_specific_force = speed * np.cos(gamma) * heading_rate
            unconstrained_bank = np.arctan2(
                lateral_specific_force, vertical_specific_force
            )
            desired_bank = float(
                np.clip(
                    unconstrained_bank,
                    -self.maximum_bank_rad,
                    self.maximum_bank_rad,
                )
            )
            bank_bound_limited = not np.isclose(
                desired_bank, unconstrained_bank, rtol=0.0, atol=1e-12
            )
            bank, bank_rate_limited = _bounded_rate(
                desired_bank,
                None if previous_control is None else previous_control.bank_angle_rad,
                self.maximum_bank_rate_rad_s,
                command_step_s,
            )
            required_lift_n = mass * max(vertical_specific_force, 0.0) / np.cos(bank)
            target_lift_coefficient = required_lift_n / (
                dynamic_pressure * plant.wing_area_m2
            )
            desired_pitch, pitch_bound_limited = self._pitch_for_lift_coefficient(
                plant,
                target_lift_coefficient=target_lift_coefficient,
                flight_path_angle_rad=gamma,
                mach=mach,
            )
            pitch, pitch_rate_limited = _bounded_rate(
                desired_pitch,
                None if previous_control is None else previous_control.pitch_angle_rad,
                self.maximum_pitch_rate_rad_s,
                command_step_s,
            )
            zero_thrust = PitchBankThrustInput(
                pitch_angle_rad=pitch,
                bank_angle_rad=bank,
                net_thrust_n=0.0,
                static_temperature_k=static_temperature_k,
                wind_north_mps=wind_north_mps,
                wind_east_mps=wind_east_mps,
                wind_down_mps=wind_down_mps,
                drag_coefficient_increment=drag_coefficient_increment,
            )
            return (
                bank,
                pitch,
                float(target_lift_coefficient),
                plant.forces(values, zero_thrust),
                bank_bound_limited or bank_rate_limited,
                pitch_bound_limited or pitch_rate_limited,
            )

        requested_gamma_target = (
            reference_flight_path_angle_rad + altitude_gamma_correction
        )
        effective_gamma_target = requested_gamma_target
        for _ in range(3):
            _, _, _, provisional_forces, _, _ = attitude_for_gamma_target(
                effective_gamma_target
            )
            revised_gamma_target = requested_gamma_target
            if operational_thrust_target_n is not None:
                operational_energy_sine = (
                    (operational_thrust_target_n - provisional_forces.drag_n) / mass
                    - speed_acceleration
                ) / G0
                revised_gamma_target = float(
                    np.arcsin(np.clip(operational_energy_sine, -0.95, 0.95))
                )
            elif speed_guidance_mode == "CAS_MACH":
                maximum_energy_sine = (
                    (maximum_thrust - provisional_forces.drag_n) / mass
                    - speed_acceleration
                ) / G0
                maximum_energy_gamma = float(
                    np.arcsin(np.clip(maximum_energy_sine, -0.95, 0.95))
                )
                revised_gamma_target = min(requested_gamma_target, maximum_energy_gamma)
            elif speed_guidance_mode == "MACH_CAS":
                minimum_energy_sine = (
                    (minimum_thrust - provisional_forces.drag_n) / mass
                    - speed_acceleration
                ) / G0
                minimum_energy_gamma = float(
                    np.arcsin(np.clip(minimum_energy_sine, -0.95, 0.95))
                )
                revised_gamma_target = max(requested_gamma_target, minimum_energy_gamma)
            if np.isclose(
                revised_gamma_target,
                effective_gamma_target,
                rtol=0.0,
                atol=1e-12,
            ):
                break
            effective_gamma_target = revised_gamma_target
        energy_limited = speed_energy_limited or not np.isclose(
            effective_gamma_target, requested_gamma_target, rtol=0.0, atol=1e-12
        )
        bank, pitch, target_lift_coefficient, forces, bank_limited, pitch_limited = (
            attitude_for_gamma_target(effective_gamma_target)
        )
        unconstrained_thrust = forces.drag_n + mass * (
            speed_acceleration + G0 * np.sin(effective_gamma_target)
        )
        desired_thrust = float(
            np.clip(
                unconstrained_thrust,
                minimum_thrust,
                maximum_thrust,
            )
        )
        thrust_bound_limited = not np.isclose(
            desired_thrust, unconstrained_thrust, rtol=0.0, atol=1e-9
        )
        thrust, thrust_rate_limited = _bounded_rate(
            desired_thrust,
            None if previous_control is None else previous_control.net_thrust_n,
            self.maximum_thrust_rate_n_s,
            command_step_s,
        )
        envelope_bounded_thrust = float(np.clip(thrust, minimum_thrust, maximum_thrust))
        thrust_envelope_limited = not np.isclose(
            envelope_bounded_thrust, thrust, rtol=0.0, atol=1e-9
        )
        thrust = envelope_bounded_thrust
        control = PitchBankThrustInput(
            pitch_angle_rad=pitch,
            bank_angle_rad=bank,
            net_thrust_n=thrust,
            static_temperature_k=static_temperature_k,
            wind_north_mps=wind_north_mps,
            wind_east_mps=wind_east_mps,
            wind_down_mps=wind_down_mps,
            drag_coefficient_increment=drag_coefficient_increment,
        )
        achieved_lift_coefficient = plant.forces(values, control).lift_coefficient
        return PitchBankThrustCommand(
            control=control,
            effective_reference_flight_path_angle_rad=float(
                effective_gamma_target - altitude_gamma_correction
            ),
            target_lift_coefficient=float(target_lift_coefficient),
            achieved_lift_coefficient=float(achieved_lift_coefficient),
            energy_limited=energy_limited,
            pitch_limited=pitch_limited,
            bank_limited=bank_limited,
            thrust_limited=(
                thrust_bound_limited or thrust_rate_limited or thrust_envelope_limited
            ),
        )


__all__ = ["InverseFlightMechanicsController", "PitchBankThrustCommand"]
