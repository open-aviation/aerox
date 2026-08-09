"""Bounded reference tracking above the attitude/realized-thrust plant.

This module deliberately commands actual pitch, bank, and total realized net
thrust.  It is a model-based trajectory-generation layer, not an Airbus flight
control law, surface allocator, FADEC model, or identified autopilot.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox.models.atmosphere import G0, atmosphere_at_pressure_altitude
from aerox.modeling.dynamics import (
    PitchAwareLiftModel,
    PitchBankThrustInput,
    PointMassModel,
    PointMassState,
    PointMassTrajectory,
)


FloatArray = NDArray[np.float64]


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
class TrajectoryReference:
    """Node-aligned state reference and analytic first derivatives."""

    time_s: FloatArray
    true_airspeed_mps: FloatArray
    true_airspeed_rate_mps2: FloatArray
    flight_path_angle_rad: FloatArray
    flight_path_angle_rate_rad_s: FloatArray
    heading_rad: FloatArray
    heading_rate_rad_s: FloatArray
    pressure_altitude_m: FloatArray

    def __post_init__(self) -> None:
        time = np.asarray(self.time_s, dtype=float)
        arrays = {
            "true_airspeed_mps": self.true_airspeed_mps,
            "true_airspeed_rate_mps2": self.true_airspeed_rate_mps2,
            "flight_path_angle_rad": self.flight_path_angle_rad,
            "flight_path_angle_rate_rad_s": self.flight_path_angle_rate_rad_s,
            "heading_rad": self.heading_rad,
            "heading_rate_rad_s": self.heading_rate_rad_s,
            "pressure_altitude_m": self.pressure_altitude_m,
        }
        if (
            time.ndim != 1
            or len(time) < 2
            or np.any(~np.isfinite(time))
            or np.any(np.diff(time) <= 0.0)
        ):
            raise ValueError("time_s must be a finite increasing vector")
        object.__setattr__(self, "time_s", time)
        for name, values in arrays.items():
            array = np.asarray(values, dtype=float)
            if array.shape != time.shape or np.any(~np.isfinite(array)):
                raise ValueError(f"{name} must be a finite vector matching time_s")
            object.__setattr__(self, name, array)
        if np.any(self.true_airspeed_mps <= 1.0):
            raise ValueError("reference true airspeed must exceed 1 m/s")


@dataclass(frozen=True)
class PitchBankThrustCommand:
    """One generated command plus inversion and saturation diagnostics."""

    control: PitchBankThrustInput
    target_lift_coefficient: float
    achieved_lift_coefficient: float
    pitch_limited: bool
    bank_limited: bool
    thrust_limited: bool


@dataclass(frozen=True)
class TrajectoryTrackingController:
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
        static_temperature_k: float,
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
        if not np.isfinite(command_step_s) or command_step_s <= 0.0:
            raise ValueError("command_step_s must be finite and positive")
        speed = values.true_airspeed_mps
        gamma = values.flight_path_angle_rad
        heading = values.heading_rad
        altitude = values.pressure_altitude_m
        mass = values.mass_kg
        speed_acceleration = reference_true_airspeed_rate_mps2 + (
            self.speed_error_gain_per_s * (reference_true_airspeed_mps - speed)
        )
        altitude_gamma_correction = self.altitude_to_flight_path_gain_rad_per_m * (
            reference_pressure_altitude_m - altitude
        )
        gamma_rate = reference_flight_path_angle_rate_rad_s + (
            self.flight_path_angle_error_gain_per_s
            * (reference_flight_path_angle_rad + altitude_gamma_correction - gamma)
        )
        heading_rate = reference_heading_rate_rad_s + self.heading_error_gain_per_s * (
            _wrapped_angle_error(reference_heading_rad, heading)
        )

        vertical_specific_force = speed * gamma_rate + G0 * np.cos(gamma)
        lateral_specific_force = speed * np.cos(gamma) * heading_rate
        desired_bank = float(
            np.clip(
                np.arctan2(lateral_specific_force, vertical_specific_force),
                -self.maximum_bank_rad,
                self.maximum_bank_rad,
            )
        )
        bank_bound_limited = not np.isclose(
            desired_bank,
            np.arctan2(lateral_specific_force, vertical_specific_force),
            rtol=0.0,
            atol=1e-12,
        )
        bank, bank_rate_limited = _bounded_rate(
            desired_bank,
            None if previous_control is None else previous_control.bank_angle_rad,
            self.maximum_bank_rate_rad_s,
            command_step_s,
        )

        atmosphere = atmosphere_at_pressure_altitude(
            np.array([altitude]), np.array([static_temperature_k])
        )
        mach = speed / float(atmosphere.speed_of_sound_mps[0])
        dynamic_pressure = 0.5 * float(atmosphere.density_kg_m3[0]) * speed**2
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
        )
        forces = plant.forces(values, zero_thrust)
        unconstrained_thrust = forces.drag_n + mass * (
            speed_acceleration + G0 * np.sin(gamma)
        )
        desired_thrust = float(
            np.clip(
                unconstrained_thrust,
                self.minimum_thrust_n,
                self.maximum_thrust_n,
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
        control = PitchBankThrustInput(
            pitch_angle_rad=pitch,
            bank_angle_rad=bank,
            net_thrust_n=thrust,
            static_temperature_k=static_temperature_k,
            wind_north_mps=wind_north_mps,
            wind_east_mps=wind_east_mps,
            wind_down_mps=wind_down_mps,
        )
        achieved_lift_coefficient = plant.forces(values, control).lift_coefficient
        return PitchBankThrustCommand(
            control=control,
            target_lift_coefficient=float(target_lift_coefficient),
            achieved_lift_coefficient=float(achieved_lift_coefficient),
            pitch_limited=pitch_bound_limited or pitch_rate_limited,
            bank_limited=bank_bound_limited or bank_rate_limited,
            thrust_limited=thrust_bound_limited or thrust_rate_limited,
        )


@dataclass(frozen=True)
class TrajectoryTrackingResult:
    """Closed-loop generated trajectory and command diagnostics."""

    trajectory: PointMassTrajectory
    reference: TrajectoryReference
    target_lift_coefficient: FloatArray
    pitch_limited: NDArray[np.bool_]
    bank_limited: NDArray[np.bool_]
    thrust_limited: NDArray[np.bool_]


def simulate_trajectory_tracking_zoh(
    *,
    plant: PointMassModel,
    controller: TrajectoryTrackingController,
    reference: TrajectoryReference,
    static_temperature_k: ArrayLike,
    initial_mass_kg: float,
    initial_true_airspeed_mps: float | None = None,
    initial_flight_path_angle_rad: float | None = None,
    initial_heading_rad: float | None = None,
    initial_pressure_altitude_m: float | None = None,
    wind_north_mps: ArrayLike | None = None,
    wind_east_mps: ArrayLike | None = None,
    wind_down_mps: ArrayLike | None = None,
    initial_north_position_m: float = 0.0,
    initial_east_position_m: float = 0.0,
    integration_rate_hz: int = 10,
    guidance_plant: PointMassModel | None = None,
    guidance_mass_scale: float = 1.0,
) -> TrajectoryTrackingResult:
    """Generate and integrate bounded controls at the reference sample rate.

    ``guidance_plant`` defaults to the propagated ``plant``. Supplying a
    distinct model permits controlled model-mismatch studies while preserving
    one physical propagated plant. ``guidance_mass_scale`` changes only the
    mass observed by the guidance calculation; the propagated mass is not
    altered.
    """

    time = reference.time_s
    temperature = np.broadcast_to(
        np.asarray(static_temperature_k, dtype=float), time.shape
    )
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError(
            "static_temperature_k must broadcast to finite positive values"
        )
    if not np.isfinite(guidance_mass_scale) or guidance_mass_scale <= 0.0:
        raise ValueError("guidance_mass_scale must be finite and positive")
    command_plant = plant if guidance_plant is None else guidance_plant
    zeros = np.zeros_like(time)

    def wind(values: ArrayLike | None, name: str) -> FloatArray:
        array = np.broadcast_to(
            zeros if values is None else np.asarray(values, dtype=float), time.shape
        ).astype(float, copy=False)
        if np.any(~np.isfinite(array)):
            raise ValueError(f"{name} must broadcast to finite values")
        return array

    north_wind = wind(wind_north_mps, "wind_north_mps")
    east_wind = wind(wind_east_mps, "wind_east_mps")
    down_wind = wind(wind_down_mps, "wind_down_mps")
    state = PointMassState(
        true_airspeed_mps=(
            reference.true_airspeed_mps[0]
            if initial_true_airspeed_mps is None
            else initial_true_airspeed_mps
        ),
        flight_path_angle_rad=(
            reference.flight_path_angle_rad[0]
            if initial_flight_path_angle_rad is None
            else initial_flight_path_angle_rad
        ),
        heading_rad=(
            reference.heading_rad[0]
            if initial_heading_rad is None
            else initial_heading_rad
        ),
        north_position_m=initial_north_position_m,
        east_position_m=initial_east_position_m,
        pressure_altitude_m=(
            reference.pressure_altitude_m[0]
            if initial_pressure_altitude_m is None
            else initial_pressure_altitude_m
        ),
        mass_kg=initial_mass_kg,
    )
    states = np.empty((len(time), 7), dtype=float)
    pitch = np.empty(len(time), dtype=float)
    bank = np.empty(len(time), dtype=float)
    thrust = np.empty(len(time), dtype=float)
    target_cl = np.empty(len(time), dtype=float)
    pitch_limited = np.empty(len(time), dtype=bool)
    bank_limited = np.empty(len(time), dtype=bool)
    thrust_limited = np.empty(len(time), dtype=bool)
    angle = np.empty(len(time), dtype=float)
    lift_coefficient = np.empty(len(time), dtype=float)
    lift = np.empty(len(time), dtype=float)
    drag = np.empty(len(time), dtype=float)
    fuel_flow = np.empty(len(time), dtype=float)
    previous: PitchBankThrustInput | None = None

    for index in range(len(time)):
        states[index] = state.to_array()
        step_s = 1.0 if index == 0 else float(time[index] - time[index - 1])
        guidance_state = replace(
            state,
            mass_kg=state.mass_kg * guidance_mass_scale,
        )
        result = controller.command(
            plant=command_plant,
            state=guidance_state,
            reference_true_airspeed_mps=float(reference.true_airspeed_mps[index]),
            reference_true_airspeed_rate_mps2=float(
                reference.true_airspeed_rate_mps2[index]
            ),
            reference_flight_path_angle_rad=float(
                reference.flight_path_angle_rad[index]
            ),
            reference_flight_path_angle_rate_rad_s=float(
                reference.flight_path_angle_rate_rad_s[index]
            ),
            reference_heading_rad=float(reference.heading_rad[index]),
            reference_heading_rate_rad_s=float(reference.heading_rate_rad_s[index]),
            reference_pressure_altitude_m=float(reference.pressure_altitude_m[index]),
            static_temperature_k=float(temperature[index]),
            previous_control=previous,
            command_step_s=step_s,
            wind_north_mps=float(north_wind[index]),
            wind_east_mps=float(east_wind[index]),
            wind_down_mps=float(down_wind[index]),
        )
        control = result.control
        previous = control
        pitch[index] = control.pitch_angle_rad
        bank[index] = control.bank_angle_rad
        thrust[index] = control.net_thrust_n
        target_cl[index] = result.target_lift_coefficient
        pitch_limited[index] = result.pitch_limited
        bank_limited[index] = result.bank_limited
        thrust_limited[index] = result.thrust_limited
        forces = plant.forces(state, control)
        angle[index] = forces.angle_of_attack_rad
        lift_coefficient[index] = forces.lift_coefficient
        lift[index] = forces.lift_n
        drag[index] = forces.drag_n
        fuel_flow[index] = forces.fuel_flow_kg_s
        if index == len(time) - 1:
            break
        interval = float(time[index + 1] - time[index])
        state = plant.integrate_zoh_step(
            state,
            control,
            interval,
            integration_rate_hz=integration_rate_hz,
        )

    trajectory = PointMassTrajectory(
        time_s=time,
        true_airspeed_mps=states[:, 0],
        flight_path_angle_rad=states[:, 1],
        heading_rad=states[:, 2],
        north_position_m=states[:, 3],
        east_position_m=states[:, 4],
        pressure_altitude_m=states[:, 5],
        mass_kg=states[:, 6],
        pitch_angle_rad=pitch,
        bank_angle_rad=bank,
        angle_of_attack_rad=angle,
        net_thrust_n=thrust,
        lift_coefficient=lift_coefficient,
        lift_n=lift,
        drag_n=drag,
        fuel_flow_kg_s=fuel_flow,
    )
    return TrajectoryTrackingResult(
        trajectory=trajectory,
        reference=reference,
        target_lift_coefficient=target_cl,
        pitch_limited=pitch_limited,
        bank_limited=bank_limited,
        thrust_limited=thrust_limited,
    )
