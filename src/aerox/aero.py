"""Aeronautical atmosphere and airspeed calculations.

This NumPy implementation is ported from ``openap.aero`` at commit
``472ab3e48d52ae62f9586d2e75e143ea4182a9e9`` (OpenAP 2.6.0).  Aerox keeps
the established OpenAP equations and public function names so air-data logic
does not diverge between the two projects.  OpenAP's optional CasADi/JAX
backend abstraction and legacy geographic re-exports are intentionally not
copied into this dependency-light package.

OpenAP is licensed under GNU LGPL v3.  The upstream source is available at
https://github.com/junzis/openap/blob/472ab3e48d52ae62f9586d2e75e143ea4182a9e9/openap/aero.py
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Unit conversion and ISA constants retained from OpenAP.
kts = 0.514444  # knot -> m/s
ft = 0.3048  # ft -> m
fpm = 0.00508  # ft/min -> m/s
inch = 0.0254  # inch -> m
sqft = 0.09290304  # square foot -> m2
nm = 1852.0  # nautical mile -> m
lbs = 0.453592  # pound -> kg
g0 = 9.80665  # m/s2
R = 287.05287  # m2/(s2 K)
p0 = 101325.0  # Pa
rho0 = 1.225  # kg/m3
T0 = 288.15  # K
gamma = 1.40  # cp/cv
gamma1 = 0.2  # (gamma - 1) / 2
gamma2 = 3.5  # gamma / (gamma - 1)
beta = -0.0065  # K/m below tropopause
r_earth = 6371000.0  # m
a0 = 340.293988  # m/s, sqrt(gamma * R * T0)


class Aero:
    """NumPy aeronautical atmosphere and airspeed calculations."""

    kts = kts
    ft = ft
    fpm = fpm
    inch = inch
    sqft = sqft
    nm = nm
    lbs = lbs
    g0 = g0
    R = R
    p0 = p0
    rho0 = rho0
    T0 = T0
    gamma = gamma
    gamma1 = gamma1
    gamma2 = gamma2
    beta = beta
    r_earth = r_earth
    a0 = a0

    def atmos(self, h: Any, dT: Any = 0) -> tuple[Any, Any, Any]:
        """Compute pressure, density, and temperature at altitude in metres."""
        altitude = np.asarray(h)
        temperature_shift = np.clip(dT, -25, 15)
        shifted_sea_level_temperature = T0 + temperature_shift
        temperature = np.maximum(
            shifted_sea_level_temperature + beta * altitude,
            216.65 + temperature_shift,
        )
        height_above_tropopause = np.maximum(0.0, altitude - 11000.0)
        tropospheric_density = rho0 * np.power(
            temperature / shifted_sea_level_temperature,
            4.256848030018761,
        )
        density_value = tropospheric_density * np.exp(
            -height_above_tropopause / 6341.552161
        )
        pressure_value = density_value * R * temperature
        return pressure_value, density_value, temperature

    def temperature(self, h: Any, dT: Any = 0) -> Any:
        """Compute ISA temperature in kelvin."""
        return self.atmos(h, dT=dT)[2]

    def pressure(self, h: Any, dT: Any = 0) -> Any:
        """Compute ISA pressure in pascals."""
        return self.atmos(h, dT=dT)[0]

    def density(self, h: Any, dT: Any = 0) -> Any:
        """Compute ISA density in kg/m3."""
        return self.atmos(h, dT=dT)[1]

    def vsound(self, h: Any, dT: Any = 0) -> Any:
        """Compute speed of sound in m/s."""
        return np.sqrt(gamma * R * self.temperature(h, dT=dT))

    def viscosity(self, h: Any, dT: Any = 0) -> Any:
        """Compute dynamic viscosity in Pa s, with Sutherland's law."""
        temperature = self.temperature(h, dT=dT)
        return 1.458e-6 * temperature**1.5 / (temperature + 110.4)

    def reynolds_per_m(self, mach: Any, h: Any, dT: Any = 0) -> Any:
        """Compute the Reynolds number per metre at a Mach number."""
        _, density_value, _ = self.atmos(h, dT=dT)
        v_tas = self.mach2tas(mach, h, dT=dT)
        return density_value * v_tas / self.viscosity(h, dT=dT)

    def h_isa(self, p: Any, dT: Any = 0) -> Any:
        """Compute ISA altitude in metres for pressure in pascals."""
        pressure_value = np.asarray(p)
        shifted_sea_level_temperature = T0 + dT
        temperature_value = shifted_sea_level_temperature * np.power(
            p0 / pressure_value, (beta * R) / g0
        )
        troposphere_height = (temperature_value - shifted_sea_level_temperature) / beta
        tropopause_temperature = shifted_sea_level_temperature + beta * 11000
        stratosphere_height = (
            -R * tropopause_temperature / g0 * np.log(pressure_value / 22630) + 11000
        )
        return np.where(pressure_value > 22630, troposphere_height, stratosphere_height)

    def tas2mach(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed in m/s to Mach number."""
        return np.asarray(v_tas) / self.vsound(h, dT=dT)

    def mach2tas(self, mach: Any, h: Any, dT: Any = 0) -> Any:
        """Convert Mach number to true airspeed in m/s."""
        return np.asarray(mach) * self.vsound(h, dT=dT)

    def eas2tas(self, v_eas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert equivalent airspeed in m/s to true airspeed in m/s."""
        density_value = self.density(h, dT=dT)
        return np.asarray(v_eas) * np.sqrt(rho0 / density_value)

    def tas2eas(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed in m/s to equivalent airspeed in m/s."""
        density_value = self.density(h, dT=dT)
        return np.asarray(v_tas) * np.sqrt(density_value / rho0)

    def cas2tas(self, v_cas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert calibrated airspeed in m/s to true airspeed in m/s."""
        pressure_value, density_value, _ = self.atmos(h, dT=dT)
        cas = np.asarray(v_cas)
        impact_pressure = p0 * (
            np.power(1.0 + rho0 * cas * cas / (7.0 * p0), 3.5) - 1.0
        )
        return np.sqrt(
            7.0
            * pressure_value
            / density_value
            * (np.power(1.0 + impact_pressure / pressure_value, 2.0 / 7.0) - 1.0)
        )

    def tas2cas(self, v_tas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert true airspeed in m/s to calibrated airspeed in m/s."""
        pressure_value, density_value, _ = self.atmos(h, dT=dT)
        tas = np.asarray(v_tas)
        impact_pressure = pressure_value * (
            np.power(1.0 + density_value * tas * tas / (7.0 * pressure_value), 3.5)
            - 1.0
        )
        return np.sqrt(
            7.0 * p0 / rho0 * (np.power(impact_pressure / p0 + 1.0, 2.0 / 7.0) - 1.0)
        )

    def mach2cas(self, mach: Any, h: Any, dT: Any = 0) -> Any:
        """Convert Mach number to calibrated airspeed in m/s."""
        return self.tas2cas(self.mach2tas(mach, h, dT=dT), h, dT=dT)

    def cas2mach(self, v_cas: Any, h: Any, dT: Any = 0) -> Any:
        """Convert calibrated airspeed in m/s to Mach number."""
        return self.tas2mach(self.cas2tas(v_cas, h, dT=dT), h, dT=dT)

    def crossover_alt(self, v_cas: Any, mach: Any, dT: Any = 0) -> Any:
        """Compute constant-CAS/constant-Mach crossover altitude in metres."""
        mach_safe = np.maximum(mach, 1e-4)
        pressure_ratio = (
            np.power(0.2 * (np.asarray(v_cas) / a0) ** 2 + 1, 3.5) - 1
        ) / (np.power(0.2 * mach_safe**2 + 1, 3.5) - 1)
        shifted_sea_level_temperature = T0 + dT
        return (
            shifted_sea_level_temperature
            / beta
            * (np.power(pressure_ratio, -R * beta / g0) - 1)
        )


_default_aero = Aero()


def atmos(h: Any, dT: Any = 0) -> tuple[Any, Any, Any]:
    return _default_aero.atmos(h, dT)


def temperature(h: Any, dT: Any = 0) -> Any:
    return _default_aero.temperature(h, dT)


def pressure(h: Any, dT: Any = 0) -> Any:
    return _default_aero.pressure(h, dT)


def density(h: Any, dT: Any = 0) -> Any:
    return _default_aero.density(h, dT)


def vsound(h: Any, dT: Any = 0) -> Any:
    return _default_aero.vsound(h, dT)


def viscosity(h: Any, dT: Any = 0) -> Any:
    return _default_aero.viscosity(h, dT)


def reynolds_per_m(mach: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.reynolds_per_m(mach, h, dT)


def h_isa(p: Any, dT: Any = 0) -> Any:
    return _default_aero.h_isa(p, dT)


def tas2mach(v_tas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.tas2mach(v_tas, h, dT)


def mach2tas(mach: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.mach2tas(mach, h, dT)


def eas2tas(v_eas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.eas2tas(v_eas, h, dT)


def tas2eas(v_tas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.tas2eas(v_tas, h, dT)


def cas2tas(v_cas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.cas2tas(v_cas, h, dT)


def tas2cas(v_tas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.tas2cas(v_tas, h, dT)


def mach2cas(mach: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.mach2cas(mach, h, dT)


def cas2mach(v_cas: Any, h: Any, dT: Any = 0) -> Any:
    return _default_aero.cas2mach(v_cas, h, dT)


def crossover_alt(v_cas: Any, mach: Any, dT: Any = 0) -> Any:
    return _default_aero.crossover_alt(v_cas, mach, dT)
