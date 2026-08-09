"""Array validation and force-model adapters around :mod:`aerox.aero`.

Pressure altitude is an ISA pressure coordinate.  Keeping pressure recovery
separate from temperature lets QAR analyses compute dynamic pressure from CAS
without silently depending on an unverified weather-temperature product.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox import aero

FloatArray = NDArray[np.float64]

G0 = aero.g0
DRY_AIR_GAS_CONSTANT = aero.R
HEAT_CAPACITY_RATIO = aero.gamma
SEA_LEVEL_TEMPERATURE_K = aero.T0
SEA_LEVEL_PRESSURE_PA = aero.p0
TROPOPAUSE_ALTITUDE_M = 11_000.0
TROPOSPHERE_LAPSE_RATE_K_PER_M = -0.0065
MIN_SUPPORTED_ALTITUDE_M = -2_000.0
MAX_SUPPORTED_ALTITUDE_M = 20_000.0
SUTHERLAND_VISCOSITY_COEFFICIENT = 1.458e-6
SUTHERLAND_TEMPERATURE_K = 110.4


@dataclass(frozen=True)
class AtmosphereState:
    """Atmospheric properties at a pressure altitude."""

    pressure_pa: FloatArray
    temperature_k: FloatArray
    density_kg_m3: FloatArray
    speed_of_sound_mps: FloatArray


def _as_float_array(values: ArrayLike) -> FloatArray:
    return np.asarray(values, dtype=float)


def isa_atmosphere(pressure_altitude_m: ArrayLike) -> AtmosphereState:
    """Return the ISA state from -2 km through the 20 km isothermal layer.

    The input is geopotential pressure altitude, matching the coordinate used
    by a pressure-altitude channel.  No geometric/geopotential conversion is
    applied.
    """
    altitude = _as_float_array(pressure_altitude_m)
    if np.any(~np.isfinite(altitude)):
        raise ValueError("pressure_altitude_m must be finite")
    if np.any(
        (altitude < MIN_SUPPORTED_ALTITUDE_M) | (altitude > MAX_SUPPORTED_ALTITUDE_M)
    ):
        raise ValueError(
            f"pressure altitude must be within {MIN_SUPPORTED_ALTITUDE_M:g} "
            f"and {MAX_SUPPORTED_ALTITUDE_M:g} m"
        )

    pressure, density, temperature = aero.atmos(altitude)
    speed_of_sound = aero.vsound(altitude)
    return AtmosphereState(
        pressure_pa=pressure,
        temperature_k=temperature,
        density_kg_m3=density,
        speed_of_sound_mps=speed_of_sound,
    )


def atmosphere_at_pressure_altitude(
    pressure_altitude_m: ArrayLike,
    static_temperature_k: ArrayLike,
) -> AtmosphereState:
    """Combine ISA pressure with an independently supplied static temperature."""
    standard = isa_atmosphere(pressure_altitude_m)
    temperature = np.broadcast_to(
        _as_float_array(static_temperature_k), standard.pressure_pa.shape
    ).astype(float, copy=False)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("static_temperature_k must be finite and positive")
    density = standard.pressure_pa / (DRY_AIR_GAS_CONSTANT * temperature)
    speed_of_sound = np.sqrt(HEAT_CAPACITY_RATIO * DRY_AIR_GAS_CONSTANT * temperature)
    return AtmosphereState(
        pressure_pa=standard.pressure_pa,
        temperature_k=temperature,
        density_kg_m3=density,
        speed_of_sound_mps=speed_of_sound,
    )


def dynamic_viscosity_air(static_temperature_k: ArrayLike) -> FloatArray:
    """Return dry-air dynamic viscosity in Pa s using Sutherland's law."""

    temperature = _as_float_array(static_temperature_k)
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0.0):
        raise ValueError("static_temperature_k must be finite and positive")
    return (
        SUTHERLAND_VISCOSITY_COEFFICIENT
        * temperature**1.5
        / (temperature + SUTHERLAND_TEMPERATURE_K)
    )


def unit_reynolds_number(
    density_kg_m3: ArrayLike,
    true_airspeed_mps: ArrayLike,
    static_temperature_k: ArrayLike,
) -> FloatArray:
    """Return Reynolds number per metre, ``rho * TAS / mu``."""

    density, speed, temperature = np.broadcast_arrays(
        _as_float_array(density_kg_m3),
        _as_float_array(true_airspeed_mps),
        _as_float_array(static_temperature_k),
    )
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise ValueError("density_kg_m3 must be finite and positive")
    if np.any(~np.isfinite(speed)) or np.any(speed <= 0.0):
        raise ValueError("true_airspeed_mps must be finite and positive")
    return density * speed / dynamic_viscosity_air(temperature)


def impact_pressure_from_calibrated_airspeed(
    calibrated_airspeed_mps: ArrayLike,
) -> FloatArray:
    """Return subsonic pitot impact pressure represented by CAS."""
    speed = _as_float_array(calibrated_airspeed_mps)
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("calibrated_airspeed_mps must be finite and non-negative")
    return aero.p0 * (np.power(1.0 + aero.rho0 * speed**2 / (7.0 * aero.p0), 3.5) - 1.0)


def mach_from_calibrated_airspeed(
    calibrated_airspeed_mps: ArrayLike,
    pressure_altitude_m: ArrayLike,
) -> FloatArray:
    """Recover subsonic Mach using the canonical OpenAP CAS conversion."""
    speed = _as_float_array(calibrated_airspeed_mps)
    altitude = _as_float_array(pressure_altitude_m)
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("calibrated_airspeed_mps must be finite and non-negative")
    if np.any(~np.isfinite(altitude)):
        raise ValueError("pressure_altitude_m must be finite")
    return np.asarray(aero.cas2mach(speed, altitude), dtype=float)


def calibrated_airspeed_from_mach(
    mach: ArrayLike,
    pressure_altitude_m: ArrayLike,
) -> FloatArray:
    """Return CAS from Mach while treating altitude as a pressure coordinate."""

    mach_value = _as_float_array(mach)
    altitude = _as_float_array(pressure_altitude_m)
    if np.any(~np.isfinite(mach_value)) or np.any(mach_value < 0):
        raise ValueError("mach must be finite and non-negative")
    standard = isa_atmosphere(altitude)
    impact_pressure = standard.pressure_pa * (
        np.power(1.0 + 0.2 * mach_value**2, 3.5) - 1.0
    )
    return np.sqrt(
        7.0
        * SEA_LEVEL_PRESSURE_PA
        / aero.rho0
        * (np.power(impact_pressure / SEA_LEVEL_PRESSURE_PA + 1.0, 2.0 / 7.0) - 1.0)
    )


def mach_from_true_airspeed(
    true_airspeed_mps: ArrayLike,
    pressure_altitude_m: ArrayLike,
    static_temperature_k: ArrayLike,
) -> FloatArray:
    """Return Mach from TAS, pressure altitude, and independent temperature."""

    speed = _as_float_array(true_airspeed_mps)
    if np.any(~np.isfinite(speed)) or np.any(speed < 0):
        raise ValueError("true_airspeed_mps must be finite and non-negative")
    atmosphere = atmosphere_at_pressure_altitude(
        pressure_altitude_m, static_temperature_k
    )
    return speed / atmosphere.speed_of_sound_mps


def true_airspeed_from_mach(
    mach: ArrayLike,
    pressure_altitude_m: ArrayLike,
    static_temperature_k: ArrayLike,
) -> FloatArray:
    """Return TAS from Mach at pressure altitude and forecast temperature."""

    mach_value = _as_float_array(mach)
    if np.any(~np.isfinite(mach_value)) or np.any(mach_value < 0):
        raise ValueError("mach must be finite and non-negative")
    atmosphere = atmosphere_at_pressure_altitude(
        pressure_altitude_m, static_temperature_k
    )
    return mach_value * atmosphere.speed_of_sound_mps


def calibrated_airspeed_from_true_airspeed(
    true_airspeed_mps: ArrayLike,
    pressure_altitude_m: ArrayLike,
    static_temperature_k: ArrayLike,
) -> FloatArray:
    """Return CAS from TAS without treating pressure altitude as geometric."""

    return calibrated_airspeed_from_mach(
        mach_from_true_airspeed(
            true_airspeed_mps,
            pressure_altitude_m,
            static_temperature_k,
        ),
        pressure_altitude_m,
    )


def true_airspeed_from_calibrated_airspeed(
    calibrated_airspeed_mps: ArrayLike,
    pressure_altitude_m: ArrayLike,
    static_temperature_k: ArrayLike,
) -> FloatArray:
    """Return TAS from CAS at pressure altitude and forecast temperature."""

    mach = mach_from_calibrated_airspeed(calibrated_airspeed_mps, pressure_altitude_m)
    return true_airspeed_from_mach(
        mach,
        pressure_altitude_m,
        static_temperature_k,
    )


def dynamic_pressure_from_calibrated_airspeed(
    calibrated_airspeed_mps: ArrayLike,
    pressure_altitude_m: ArrayLike,
) -> FloatArray:
    """Return ``0.5 * rho * TAS**2`` from CAS and pressure altitude.

    This form uses the isentropic air-data relation to recover Mach and does
    not need temperature as an input.
    """
    altitude = _as_float_array(pressure_altitude_m)
    pressure = np.asarray(aero.pressure(altitude), dtype=float)
    mach = mach_from_calibrated_airspeed(calibrated_airspeed_mps, altitude)
    return 0.5 * HEAT_CAPACITY_RATIO * pressure * mach**2
