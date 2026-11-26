"""Reduced 3D trajectory plant driven by pitch, bank, and realized net thrust.

This is the primary intermediate-fidelity interface for recorded-flight replay.
It intentionally places surface allocation, proprietary flight-control laws,
FADEC logic, and spool dynamics outside the flight-mechanics plant.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox.models.lift import EffectiveLiftMap
from aerox.models._atmosphere import (
    G0,
    atmosphere_at_pressure_altitude,
    unit_reynolds_number,
)
from aerox.models.drag import DragCondition, DragModel
from aerox.models.openap import CalibratedFuelFlow, OpenAPFuelFlowPrior

FloatArray = NDArray[np.float64]


@runtime_checkable
class PitchAwareLiftModel(Protocol):
    """Lift model that must distinguish decoded pitch from physical gamma."""

    def angle_and_lift_coefficient_scalar(
        self, *, pitch_angle_rad: float, flight_path_angle_rad: float, mach: float
    ) -> tuple[float, float]: ...


@dataclass(frozen=True)
class PointMassState:
    """Named representation of the seven coordinated-flight plant states."""

    true_airspeed_mps: float
    flight_path_angle_rad: float
    heading_rad: float
    north_position_m: float
    east_position_m: float
    pressure_altitude_m: float
    mass_kg: float

    def __post_init__(self) -> None:
        values = self.to_array()
        if np.any(~np.isfinite(values)):
            raise ValueError("plant state values must be finite")
        if self.true_airspeed_mps <= 1.0 or self.mass_kg <= 0.0:
            raise ValueError("plant speed and mass must remain positive")
        if abs(np.cos(self.flight_path_angle_rad)) < 0.1:
            raise ValueError("flight-path angle is outside the reduced-model envelope")

    @classmethod
    def from_array(cls, state: ArrayLike) -> PointMassState:
        """Validate and name a seven-element integration vector."""

        values = np.asarray(state, dtype=float)
        if values.shape != (7,):
            raise ValueError("plant state must be a seven-element vector")
        return cls(*(float(value) for value in values))

    def to_array(self) -> FloatArray:
        """Return the state in the numerical integrator's canonical order."""

        return np.array(
            [
                self.true_airspeed_mps,
                self.flight_path_angle_rad,
                self.heading_rad,
                self.north_position_m,
                self.east_position_m,
                self.pressure_altitude_m,
                self.mass_kg,
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class PitchBankThrustInput:
    """One held attitude/thrust/configuration sample plus environmental forcing.

    ``net_thrust_n`` is total realized propulsive force resolved along the
    air-relative flight path. It excludes aerodynamic drag. Pitch and bank are
    actual attitudes, not elevator/aileron commands. The non-negative drag
    increment is an effective configuration term. It can represent deployed
    drag devices or unresolved configuration drag without assigning a specific
    physical device.
    """

    pitch_angle_rad: float
    bank_angle_rad: float
    net_thrust_n: float
    static_temperature_k: float
    wind_north_mps: float = 0.0
    wind_east_mps: float = 0.0
    wind_down_mps: float = 0.0
    drag_coefficient_increment: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.pitch_angle_rad,
            self.bank_angle_rad,
            self.net_thrust_n,
            self.static_temperature_k,
            self.wind_north_mps,
            self.wind_east_mps,
            self.wind_down_mps,
            self.drag_coefficient_increment,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("pitch-bank-thrust input values must be finite")
        if self.net_thrust_n < 0.0:
            raise ValueError("net_thrust_n must be non-negative")
        if self.drag_coefficient_increment < 0.0:
            raise ValueError("drag_coefficient_increment must be non-negative")
        if self.static_temperature_k <= 0.0:
            raise ValueError("static_temperature_k must be positive")
        if abs(self.bank_angle_rad) >= np.deg2rad(85.0):
            raise ValueError("bank angle is outside the reduced-model envelope")


@dataclass(frozen=True)
class PointMassForces:
    """Aerodynamic and propulsion diagnostics for one state/control pair."""

    angle_of_attack_rad: float
    mach: float
    lift_coefficient: float
    lift_n: float
    drag_n: float
    net_thrust_n: float
    fuel_flow_kg_s: float


@dataclass(frozen=True)
class PointMassTrajectory:
    """Node-aligned output of the point-mass trajectory plant."""

    time_s: FloatArray
    true_airspeed_mps: FloatArray
    flight_path_angle_rad: FloatArray
    heading_rad: FloatArray
    north_position_m: FloatArray
    east_position_m: FloatArray
    pressure_altitude_m: FloatArray
    mass_kg: FloatArray
    pitch_angle_rad: FloatArray
    bank_angle_rad: FloatArray
    angle_of_attack_rad: FloatArray
    net_thrust_n: FloatArray
    lift_coefficient: FloatArray
    lift_n: FloatArray
    drag_n: FloatArray
    fuel_flow_kg_s: FloatArray


@dataclass(frozen=True)
class HeadingRateReplayResult:
    """Replay-only trajectory plus the force inconsistency caused by override.

    ``implied_lateral_force_residual_n`` is resolved normal to the horizontal
    air-relative velocity. It is the force required by the observed heading
    rate minus the lateral component of modeled lift. It is not a body-axis
    side-force estimate.
    """

    trajectory: PointMassTrajectory
    observed_heading_rate_rad_s: FloatArray
    coordinated_heading_rate_rad_s: FloatArray
    implied_lateral_force_residual_n: FloatArray


@dataclass(frozen=True)
class PointMassModel:
    """Seven-state coordinated-flight plant with direct realized thrust input.

    State order is true airspeed, air-relative flight-path angle, true heading,
    local north position, local east position, pressure altitude, and mass.
    Angle of attack is approximated by pitch minus flight-path angle. Bank
    generates heading rate through the lateral component of lift.
    """

    wing_area_m2: float
    lift_map: EffectiveLiftMap | PitchAwareLiftModel
    drag_model: DragModel
    fuel_flow_prior: OpenAPFuelFlowPrior | CalibratedFuelFlow

    def __post_init__(self) -> None:
        if not np.isfinite(self.wing_area_m2) or self.wing_area_m2 <= 0.0:
            raise ValueError("wing_area_m2 must be finite and positive")

    def drag_coefficient_scalar(
        self,
        lift_coefficient: float,
        mach: float,
        *,
        reynolds_per_m: float,
        flap_angle_rad: float = 0.0,
        gear_extension_fraction: float = 0.0,
        spoiler_deflection_rad: float = 0.0,
        slat_angle_rad: float = 0.0,
    ) -> float:
        """Evaluate drag at the complete local aerodynamic condition."""

        return self.drag_model._drag_coefficient_scalar(
            lift_coefficient,
            DragCondition(
                mach=mach,
                reynolds_per_m=reynolds_per_m,
                flap_angle_rad=flap_angle_rad,
                gear_extension_fraction=gear_extension_fraction,
                spoiler_deflection_rad=spoiler_deflection_rad,
                slat_angle_rad=slat_angle_rad,
            ),
        )

    @staticmethod
    def validate_state(
        state: PointMassState | ArrayLike,
    ) -> PointMassState:
        """Return the validated named state accepted by all plant methods."""

        if isinstance(state, PointMassState):
            return state
        return PointMassState.from_array(state)

    def forces(
        self, state: PointMassState | ArrayLike, control: PitchBankThrustInput
    ) -> PointMassForces:
        """Return force diagnostics without advancing the state."""
        values = self.validate_state(state)
        speed = values.true_airspeed_mps
        gamma = values.flight_path_angle_rad
        altitude = values.pressure_altitude_m
        atmosphere = atmosphere_at_pressure_altitude(
            np.array([altitude]), np.array([control.static_temperature_k])
        )
        density = float(atmosphere.density_kg_m3[0])
        speed_of_sound = float(atmosphere.speed_of_sound_mps[0])
        mach = speed / speed_of_sound
        dynamic_pressure = 0.5 * density * speed**2
        if isinstance(self.lift_map, PitchAwareLiftModel):
            angle_of_attack, lift_coefficient = (
                self.lift_map.angle_and_lift_coefficient_scalar(
                    pitch_angle_rad=control.pitch_angle_rad,
                    flight_path_angle_rad=gamma,
                    mach=mach,
                )
            )
        else:
            angle_of_attack = control.pitch_angle_rad - gamma
            lift_coefficient = self.lift_map.lift_coefficient_scalar(
                angle_of_attack, mach
            )
        reynolds_per_metre = float(
            unit_reynolds_number(
                density,
                speed,
                control.static_temperature_k,
            )
        )
        drag_coefficient = self.drag_coefficient_scalar(
            lift_coefficient,
            mach,
            reynolds_per_m=reynolds_per_metre,
        )
        drag_coefficient += control.drag_coefficient_increment
        lift = dynamic_pressure * self.wing_area_m2 * lift_coefficient
        drag = dynamic_pressure * self.wing_area_m2 * drag_coefficient
        fuel_flow = self.fuel_flow_prior.fuel_flow_scalar(control.net_thrust_n)
        return PointMassForces(
            angle_of_attack_rad=angle_of_attack,
            mach=mach,
            lift_coefficient=lift_coefficient,
            lift_n=lift,
            drag_n=drag,
            net_thrust_n=control.net_thrust_n,
            fuel_flow_kg_s=fuel_flow,
        )

    def derivative(
        self, state: PointMassState | ArrayLike, control: PitchBankThrustInput
    ) -> FloatArray:
        """Return the continuous state derivative for one held input."""
        values = self.validate_state(state)
        speed = values.true_airspeed_mps
        gamma = values.flight_path_angle_rad
        heading = values.heading_rad
        mass = values.mass_kg
        forces = self.forces(values, control)
        cos_gamma = np.cos(gamma)
        speed_rate = (forces.net_thrust_n - forces.drag_n) / mass - G0 * np.sin(gamma)
        gamma_rate = (
            forces.lift_n * np.cos(control.bank_angle_rad) / mass - G0 * np.cos(gamma)
        ) / speed
        heading_rate = (
            forces.lift_n * np.sin(control.bank_angle_rad) / (mass * speed * cos_gamma)
        )
        horizontal_air_speed = speed * cos_gamma
        north_rate = horizontal_air_speed * np.cos(heading) + control.wind_north_mps
        east_rate = horizontal_air_speed * np.sin(heading) + control.wind_east_mps
        altitude_rate = speed * np.sin(gamma) - control.wind_down_mps
        return np.array(
            [
                speed_rate,
                gamma_rate,
                heading_rate,
                north_rate,
                east_rate,
                altitude_rate,
                -forces.fuel_flow_kg_s,
            ]
        )

    def integrate_zoh_step(
        self,
        state: PointMassState | ArrayLike,
        control: PitchBankThrustInput,
        duration_s: float,
        *,
        integration_rate_hz: int = 10,
        heading_rate_override_rad_s: float | None = None,
    ) -> PointMassState:
        """Advance one held control interval and return the named final state."""

        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if integration_rate_hz < 1:
            raise ValueError("integration_rate_hz must be a positive integer")
        if heading_rate_override_rad_s is not None and not np.isfinite(
            heading_rate_override_rad_s
        ):
            raise ValueError("heading_rate_override_rad_s must be finite")
        state_array = self.validate_state(state).to_array()
        substeps = max(1, int(np.ceil(duration_s * integration_rate_hz)))
        step_s = duration_s / substeps

        def derivative(value: FloatArray) -> FloatArray:
            result = self.derivative(value, control)
            if heading_rate_override_rad_s is not None:
                result[2] = heading_rate_override_rad_s
            return result

        for _ in range(substeps):
            state_array = _rk4_step(derivative, state_array, step_s)
        return self.validate_state(state_array)


def _rk4_step(
    derivative: Callable[[FloatArray], FloatArray], state: FloatArray, step_s: float
) -> FloatArray:
    k1 = derivative(state)
    k2 = derivative(state + 0.5 * step_s * k1)
    k3 = derivative(state + 0.5 * step_s * k2)
    k4 = derivative(state + step_s * k3)
    return state + step_s * (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
