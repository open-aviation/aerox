"""Pinned OpenAP fuel-flow and two-spool turbofan thrust models.

These dependency-light NumPy ports preserve the relevant fuel-flow and thrust
equations from OpenAP 2.6.0 at commit
``472ab3e48d52ae62f9586d2e75e143ea4182a9e9``. Source and licensing provenance
are recorded in ``agent-modeling/modeling/openap-fuel-port.md``,
``agent-modeling/modeling/openap-thrust-port.md``, and ``agent-modeling/modeling/sources.yml``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox import aero

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OpenAPFuelFlowPrior:
    """Pinned OpenAP smooth fuel-flow curve for a selected engine family."""

    engine_count: int
    rated_thrust_n_per_engine: float
    c1: float
    c2: float
    c3: float
    engine_scale: float = 1.0

    def fuel_flow_kg_s(self, total_aircraft_thrust_n: ArrayLike) -> FloatArray:
        """Predict total fuel flow using OpenAP's bounded thrust-ratio curve."""
        thrust = np.asarray(total_aircraft_thrust_n, dtype=float)
        if np.any(~np.isfinite(thrust)):
            raise ValueError("total_aircraft_thrust_n must be finite")
        ratio = (np.maximum(thrust, 0.0) / self.engine_count) / (
            self.rated_thrust_n_per_engine
        )
        limited_ratio = (
            (
                np.logaddexp(0.0, 50.0 * (ratio - 0.03))
                - np.logaddexp(0.0, 45.0 * (ratio - 1.2))
            )
            / np.logaddexp(0.0, 50.0)
        ) + 0.03
        per_engine = self.engine_scale * (
            self.c1
            - np.exp(
                -self.c2
                * (
                    limited_ratio * np.exp(self.c3 * limited_ratio)
                    - np.log(self.c1) / self.c2
                )
            )
        )
        return self.engine_count * per_engine

    def fuel_flow_scalar(self, total_aircraft_thrust_n: float) -> float:
        """Scalar fast path matching :meth:`fuel_flow_kg_s`."""
        ratio = (max(total_aircraft_thrust_n, 0.0) / self.engine_count) / (
            self.rated_thrust_n_per_engine
        )

        def softplus(value: float) -> float:
            return max(value, 0.0) + math.log1p(math.exp(-abs(value)))

        limited_ratio = (
            softplus(50.0 * (ratio - 0.03)) - softplus(45.0 * (ratio - 1.2))
        ) / softplus(50.0) + 0.03
        per_engine = self.engine_scale * (
            self.c1
            - math.exp(
                -self.c2
                * (
                    limited_ratio * math.exp(self.c3 * limited_ratio)
                    - math.log(self.c1) / self.c2
                )
            )
        )
        return self.engine_count * per_engine


@dataclass(frozen=True)
class CalibratedFuelFlow:
    """Non-negative affine correction around a pinned fuel-flow prior."""

    prior: OpenAPFuelFlowPrior
    scale: float
    offset_kg_s: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not np.isfinite(self.offset_kg_s):
            raise ValueError("offset_kg_s must be finite")

    def fuel_flow_kg_s(self, total_aircraft_thrust_n: ArrayLike) -> FloatArray:
        prior_flow = self.prior.fuel_flow_kg_s(total_aircraft_thrust_n)
        return np.maximum(0.0, self.scale * prior_flow + self.offset_kg_s)

    def fuel_flow_scalar(self, total_aircraft_thrust_n: float) -> float:
        return max(
            0.0,
            self.scale * self.prior.fuel_flow_scalar(total_aircraft_thrust_n)
            + self.offset_kg_s,
        )


@dataclass(frozen=True)
class OpenAPTwoSpoolMaximumClimbThrust:
    """OpenAP/Bartel--Young maximum-climb thrust approximation."""

    engine_bypass_ratio: float = 5.9
    engine_count: int = 2
    cruise_altitude_m: float = 11_000.0
    cruise_mach: float = 0.8
    cruise_thrust_n_per_engine: float = 22_241.0
    reference_rate_of_climb_ft_min: float = 2_500.0

    def __post_init__(self) -> None:
        values = (
            self.engine_bypass_ratio,
            self.cruise_altitude_m,
            self.cruise_mach,
            self.cruise_thrust_n_per_engine,
            self.reference_rate_of_climb_ft_min,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError("OpenAP thrust-prior parameters must be positive")
        if not isinstance(self.engine_count, int) or self.engine_count <= 0:
            raise ValueError("engine_count must be a positive integer")

    @staticmethod
    def _d_function(mach_ratio: float) -> float:
        return -0.4204 * mach_ratio + 1.0824

    @staticmethod
    def _n_function(rate_of_climb_ft_min: float) -> float:
        return 2.667e-05 * abs(rate_of_climb_ft_min) + 0.8633

    @staticmethod
    def _m_function(
        calibrated_speed_ratio: float, rate_of_climb_ft_min: float
    ) -> float:
        rate = abs(rate_of_climb_ft_min)
        return (
            -1.2043e-1 * calibrated_speed_ratio
            - 8.8889e-9 * rate**2
            + 2.4444e-5 * rate
            + 4.7379e-1
        )

    def maximum_thrust_n(
        self,
        *,
        pressure_altitude_m: float,
        true_airspeed_mps: float,
        static_temperature_k: float,
    ) -> float:
        """Return total-aircraft thrust for the pinned reference climb rate."""
        values = pressure_altitude_m, true_airspeed_mps, static_temperature_k
        if any(not np.isfinite(value) for value in values):
            raise ValueError("thrust-prior condition must be finite")
        if true_airspeed_mps <= 0.0 or static_temperature_k <= 0.0:
            raise ValueError("speed and temperature must be positive")
        isa_temperature = float(aero.temperature(pressure_altitude_m))
        temperature_shift = float(
            np.clip(static_temperature_k - isa_temperature, -25.0, 15.0)
        )
        speed = max(10.0 * aero.kts, true_airspeed_mps)
        mach = float(aero.tas2mach(speed, pressure_altitude_m, dT=temperature_shift))
        calibrated_speed = float(
            aero.tas2cas(speed, pressure_altitude_m, dT=temperature_shift)
        )
        pressure = float(aero.pressure(pressure_altitude_m, dT=temperature_shift))
        pressure_10k = float(aero.pressure(10_000.0 * aero.ft, dT=temperature_shift))
        cruise_pressure = float(
            aero.pressure(self.cruise_altitude_m, dT=temperature_shift)
        )
        reference_calibrated_speed = float(
            aero.mach2cas(
                self.cruise_mach,
                self.cruise_altitude_m,
                dT=temperature_shift,
            )
        )
        calibrated_speed_ratio = calibrated_speed / reference_calibrated_speed
        cruise_thrust = self.cruise_thrust_n_per_engine * self.engine_count
        rate = self.reference_rate_of_climb_ft_min

        d_parameter = self._d_function(mach / self.cruise_mach)
        mach_parameter = (mach / self.cruise_mach) ** (-0.11)
        ratio_high = d_parameter * np.log(pressure / cruise_pressure) + mach_parameter

        speed_parameter = calibrated_speed_ratio ** (-0.1)
        climb_parameter = self._n_function(rate)
        exponent = -0.355 * calibrated_speed_ratio + climb_parameter
        ratio_middle = speed_parameter * (pressure / cruise_pressure) ** exponent

        thrust_10k = (
            cruise_thrust
            * speed_parameter
            * (pressure_10k / cruise_pressure) ** exponent
        )
        low_parameter = self._m_function(calibrated_speed_ratio, rate)
        ratio_low = (
            low_parameter * (pressure / cruise_pressure)
            + thrust_10k / cruise_thrust
            - low_parameter * (pressure_10k / cruise_pressure)
        )
        altitude_ft = pressure_altitude_m / aero.ft
        ratio = (
            ratio_high
            if altitude_ft > 30_000.0
            else ratio_middle
            if altitude_ft > 10_000.0
            else ratio_low
        )
        thrust = float(ratio * cruise_thrust)
        if not np.isfinite(thrust) or thrust <= 0.0:
            raise ValueError("OpenAP thrust prior returned a non-positive value")
        return thrust
