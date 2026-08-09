"""Effective turbofan identification primitives.

The thrust map here is deliberately an operational grey-box model.  It uses
corrected fan speed and ambient state, but its coefficients are identified from
QAR-derived force requirements rather than claimed as engine-deck parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import ArrayLike, NDArray

from aerox import aero

FloatArray = NDArray[np.float64]

THRUST_MAP_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "quadratic_mach": (
        "constant",
        "fan_speed_centered",
        "fan_speed_centered_squared",
        "mach_centered",
        "fan_speed_mach_interaction",
    ),
    "cubic_mach": (
        "constant",
        "fan_speed_centered",
        "fan_speed_centered_squared",
        "fan_speed_centered_cubed",
        "mach_centered",
        "fan_speed_mach_interaction",
    ),
    "cubic_mach_squared": (
        "constant",
        "fan_speed_centered",
        "fan_speed_centered_squared",
        "fan_speed_centered_cubed",
        "mach_centered",
        "mach_centered_squared",
        "fan_speed_mach_interaction",
    ),
}

N1_ENVELOPE_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "altitude_linear": (
        "constant",
        "pressure_altitude_scaled",
    ),
    "altitude_quadratic": (
        "constant",
        "pressure_altitude_scaled",
        "pressure_altitude_scaled_squared",
    ),
    "altitude_mach": (
        "constant",
        "pressure_altitude_scaled",
        "pressure_altitude_scaled_squared",
        "mach_centered",
    ),
    "altitude_mach_temperature": (
        "constant",
        "pressure_altitude_scaled",
        "pressure_altitude_scaled_squared",
        "mach_centered",
        "isa_temperature_deviation_scaled",
    ),
}


class AutothrustMode(StrEnum):
    """Reduced A320 autothrust modes exposed by the surrogate interface."""

    FIXED_CLIMB = "fixed_climb"
    FIXED_IDLE = "fixed_idle"
    VARIABLE_SPEED = "variable_speed"


def n1_envelope_features(
    *,
    pressure_altitude_m: ArrayLike,
    mach: ArrayLike,
    static_temperature_k: ArrayLike,
    feature_set: str,
) -> FloatArray:
    """Construct the design matrix for an effective N1 operating envelope.

    The features are intentionally low order.  Temperature enters as deviation
    from ISA at pressure altitude, rather than absolute temperature, to avoid
    making the altitude and temperature terms needlessly collinear.
    """
    if feature_set not in N1_ENVELOPE_FEATURE_SETS:
        raise ValueError(f"unknown N1 envelope feature set: {feature_set}")
    altitude = np.asarray(pressure_altitude_m, dtype=float)
    mach_array = np.asarray(mach, dtype=float)
    temperature = np.asarray(static_temperature_k, dtype=float)
    altitude, mach_array, temperature = np.broadcast_arrays(
        altitude, mach_array, temperature
    )
    if np.any(~np.isfinite(altitude)):
        raise ValueError("pressure_altitude_m must be finite")
    if np.any(~np.isfinite(mach_array)) or np.any(mach_array < 0):
        raise ValueError("mach must be finite and non-negative")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("static_temperature_k must be finite and positive")

    altitude_scaled = altitude / 12_000.0
    isa_temperature = np.maximum(aero.T0 + aero.beta * altitude, 216.65)
    available = {
        "constant": np.ones_like(altitude_scaled),
        "pressure_altitude_scaled": altitude_scaled,
        "pressure_altitude_scaled_squared": altitude_scaled**2,
        "mach_centered": mach_array - 0.5,
        "isa_temperature_deviation_scaled": (temperature - isa_temperature) / 20.0,
    }
    return np.column_stack(
        [available[name].ravel() for name in N1_ENVELOPE_FEATURE_SETS[feature_set]]
    )


@dataclass(frozen=True)
class EffectiveN1Envelope:
    """Condition-dependent idle-to-CLB N1 envelope for a FADEC surrogate.

    This represents effective operating boundaries inferred from aircraft data;
    it is not an engine-deck schedule.  The lower and upper surfaces correspond
    to fixed-idle and fixed-climb observations.  A variable-speed demand is a
    normalized position between them.
    """

    feature_set: str
    idle_coefficients: FloatArray
    climb_coefficients: FloatArray

    def __post_init__(self) -> None:
        expected = len(N1_ENVELOPE_FEATURE_SETS.get(self.feature_set, ()))
        idle = np.asarray(self.idle_coefficients, dtype=float)
        climb = np.asarray(self.climb_coefficients, dtype=float)
        if expected == 0:
            raise ValueError(f"unknown N1 envelope feature set: {self.feature_set}")
        if idle.shape != (expected,) or climb.shape != (expected,):
            raise ValueError("N1 envelope coefficient count does not match feature set")
        if np.any(~np.isfinite(idle)) or np.any(~np.isfinite(climb)):
            raise ValueError("N1 envelope coefficients must be finite")
        object.__setattr__(self, "idle_coefficients", idle)
        object.__setattr__(self, "climb_coefficients", climb)

    def bounds_percent(
        self,
        *,
        pressure_altitude_m: ArrayLike,
        mach: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> tuple[FloatArray, FloatArray]:
        """Return effective idle and CLB N1 limits in percent."""
        design = n1_envelope_features(
            pressure_altitude_m=pressure_altitude_m,
            mach=mach,
            static_temperature_k=static_temperature_k,
            feature_set=self.feature_set,
        )
        idle = design @ self.idle_coefficients
        climb = design @ self.climb_coefficients
        if np.any(climb <= idle):
            raise ValueError("fitted climb N1 must remain above fitted idle N1")
        return idle, climb

    def climb_limit_percent(
        self,
        *,
        pressure_altitude_m: ArrayLike,
        mach: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> FloatArray:
        """Return only the independently released effective CLB boundary.

        This deliberately does not evaluate or validate the provisional idle
        surface. It is the appropriate interface when the climb boundary has
        passed its gate but the paired idle boundary has not.
        """

        design = n1_envelope_features(
            pressure_altitude_m=pressure_altitude_m,
            mach=mach,
            static_temperature_k=static_temperature_k,
            feature_set=self.feature_set,
        )
        climb = design @ self.climb_coefficients
        if np.any(~np.isfinite(climb)) or np.any(climb <= 0.0):
            raise ValueError("fitted climb N1 must be finite and positive")
        return climb

    def normalized_demand(
        self,
        *,
        n1_percent: ArrayLike,
        pressure_altitude_m: ArrayLike,
        mach: ArrayLike,
        static_temperature_k: ArrayLike,
        clip: bool = True,
    ) -> FloatArray:
        """Map observed N1 to its normalized position inside the envelope."""
        idle, climb = self.bounds_percent(
            pressure_altitude_m=pressure_altitude_m,
            mach=mach,
            static_temperature_k=static_temperature_k,
        )
        n1 = np.asarray(n1_percent, dtype=float).ravel()
        if n1.shape != idle.shape or np.any(~np.isfinite(n1)):
            raise ValueError("n1_percent must be finite and match the condition shape")
        demand = (n1 - idle) / (climb - idle)
        return np.clip(demand, 0.0, 1.0) if clip else demand

    def commanded_n1_percent(
        self,
        *,
        mode: AutothrustMode | str,
        pressure_altitude_m: float,
        mach: float,
        static_temperature_k: float,
        demand_fraction: float | None = None,
    ) -> float:
        """Return the N1 command selected by one reduced autothrust mode."""
        selected_mode = AutothrustMode(mode)
        idle, climb = self.bounds_percent(
            pressure_altitude_m=np.array([pressure_altitude_m]),
            mach=np.array([mach]),
            static_temperature_k=np.array([static_temperature_k]),
        )
        if selected_mode is AutothrustMode.FIXED_IDLE:
            return float(idle[0])
        if selected_mode is AutothrustMode.FIXED_CLIMB:
            return float(climb[0])
        if demand_fraction is None or not np.isfinite(demand_fraction):
            raise ValueError("variable-speed mode requires a finite demand_fraction")
        fraction = float(np.clip(demand_fraction, 0.0, 1.0))
        return float(idle[0] + fraction * (climb[0] - idle[0]))


@dataclass(frozen=True)
class DemonstratedClimbN1Capacity:
    """One-sided lower confidence bound on a demonstrated CLB N1 frontier.

    The coefficient ensemble is obtained by resampling complete flights.  At a
    requested condition, ``capacity_percent`` returns a low percentile across
    those fitted upper-frontier surfaces.  This is an operationally
    demonstrated planning capacity, not an engine maximum, rating schedule, or
    extrapolatable engine deck.
    """

    feature_set: str
    bootstrap_coefficients: FloatArray
    lower_confidence_quantile: float
    minimum_pressure_altitude_m: float
    maximum_pressure_altitude_m: float
    minimum_mach: float
    maximum_mach: float
    minimum_isa_temperature_deviation_k: float
    maximum_isa_temperature_deviation_k: float

    def __post_init__(self) -> None:
        expected = len(N1_ENVELOPE_FEATURE_SETS.get(self.feature_set, ()))
        coefficients = np.asarray(self.bootstrap_coefficients, dtype=float)
        if expected == 0:
            raise ValueError(f"unknown N1 envelope feature set: {self.feature_set}")
        if coefficients.ndim != 2 or coefficients.shape[1] != expected:
            raise ValueError("bootstrap coefficient count does not match feature set")
        if coefficients.shape[0] < 2 or np.any(~np.isfinite(coefficients)):
            raise ValueError(
                "at least two finite bootstrap coefficient rows are required"
            )
        if not 0.0 < self.lower_confidence_quantile < 0.5:
            raise ValueError("lower_confidence_quantile must be between zero and 0.5")
        bounds = (
            self.minimum_pressure_altitude_m,
            self.maximum_pressure_altitude_m,
            self.minimum_mach,
            self.maximum_mach,
            self.minimum_isa_temperature_deviation_k,
            self.maximum_isa_temperature_deviation_k,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("demonstrated-capacity domain bounds must be finite")
        if self.minimum_pressure_altitude_m >= self.maximum_pressure_altitude_m:
            raise ValueError("pressure-altitude domain must have positive width")
        if self.minimum_mach < 0.0 or self.minimum_mach >= self.maximum_mach:
            raise ValueError("Mach domain must be non-negative and have positive width")
        if (
            self.minimum_isa_temperature_deviation_k
            >= self.maximum_isa_temperature_deviation_k
        ):
            raise ValueError("temperature-deviation domain must have positive width")
        object.__setattr__(self, "bootstrap_coefficients", coefficients)

    def capacity_percent(
        self,
        *,
        pressure_altitude_m: ArrayLike,
        mach: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> FloatArray:
        """Return the demonstrated N1 capacity inside the fitted QAR domain."""

        altitude = np.asarray(pressure_altitude_m, dtype=float)
        mach_array = np.asarray(mach, dtype=float)
        temperature = np.asarray(static_temperature_k, dtype=float)
        altitude, mach_array, temperature = np.broadcast_arrays(
            altitude, mach_array, temperature
        )
        supported = self.support_mask(
            pressure_altitude_m=altitude,
            mach=mach_array,
            static_temperature_k=temperature,
        )
        isa_temperature = np.maximum(aero.T0 + aero.beta * altitude, 216.65)
        deviation = temperature - isa_temperature
        if np.any(~supported):
            raise ValueError(
                "condition is outside the demonstrated QAR capacity domain: "
                f"altitude=[{float(np.min(altitude))}, {float(np.max(altitude))}], "
                f"mach=[{float(np.min(mach_array))}, {float(np.max(mach_array))}], "
                f"ISA deviation=[{float(np.min(deviation))}, {float(np.max(deviation))}]"
            )
        design = n1_envelope_features(
            pressure_altitude_m=altitude,
            mach=mach_array,
            static_temperature_k=temperature,
            feature_set=self.feature_set,
        )
        predictions = design @ self.bootstrap_coefficients.T
        capacity = np.quantile(predictions, self.lower_confidence_quantile, axis=1)
        if np.any(~np.isfinite(capacity)) or np.any(capacity <= 0.0):
            raise ValueError("demonstrated N1 capacity must be finite and positive")
        return np.asarray(capacity, dtype=float)

    def support_mask(
        self,
        *,
        pressure_altitude_m: ArrayLike,
        mach: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> NDArray[np.bool_]:
        """Return which conditions lie inside every fitted support interval."""

        altitude = np.asarray(pressure_altitude_m, dtype=float)
        mach_array = np.asarray(mach, dtype=float)
        temperature = np.asarray(static_temperature_k, dtype=float)
        altitude, mach_array, temperature = np.broadcast_arrays(
            altitude, mach_array, temperature
        )
        if np.any(~np.isfinite(altitude + mach_array + temperature)):
            raise ValueError("capacity support conditions must be finite")
        isa_temperature = np.maximum(aero.T0 + aero.beta * altitude, 216.65)
        deviation = temperature - isa_temperature
        # Independent atmosphere helpers can differ at round-off level while
        # representing the same decoded QAR condition.
        altitude_tolerance_m = 1e-3
        mach_tolerance = 1e-6
        temperature_tolerance_k = 1e-3
        feature_names = N1_ENVELOPE_FEATURE_SETS[self.feature_set]
        outside = (
            altitude < self.minimum_pressure_altitude_m - altitude_tolerance_m
        ) | (altitude > self.maximum_pressure_altitude_m + altitude_tolerance_m)
        if "mach_centered" in feature_names:
            outside |= (mach_array < self.minimum_mach - mach_tolerance) | (
                mach_array > self.maximum_mach + mach_tolerance
            )
        if "isa_temperature_deviation_scaled" in feature_names:
            outside |= (
                deviation
                < self.minimum_isa_temperature_deviation_k - temperature_tolerance_k
            ) | (
                deviation
                > self.maximum_isa_temperature_deviation_k + temperature_tolerance_k
            )
        return np.asarray(~outside, dtype=bool)


@dataclass(frozen=True)
class AsymmetricFirstOrderN1Actuator:
    """Bounded effective N1 actuator driven by a commanded fan speed.

    This is an interface model, not a CFM56 engine-deck claim.  Separate rise
    and fall constants let identification retain the most basic turbofan
    asymmetry without inventing unobserved internal spool states.  No
    aircraft-specific constants are supplied here: callers must provide values
    with their own provenance and uncertainty.

    Commands are saturated to the declared N1 envelope.  State values outside
    that envelope are rejected because silently clipping a simulated state can
    conceal integration or initialization errors.
    """

    rise_time_constant_s: float
    fall_time_constant_s: float
    minimum_n1_percent: float = 0.0
    maximum_n1_percent: float = 110.0

    def __post_init__(self) -> None:
        values = (
            self.rise_time_constant_s,
            self.fall_time_constant_s,
            self.minimum_n1_percent,
            self.maximum_n1_percent,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("N1 actuator parameters must be finite")
        if self.rise_time_constant_s <= 0 or self.fall_time_constant_s <= 0:
            raise ValueError("N1 actuator time constants must be positive")
        if self.minimum_n1_percent >= self.maximum_n1_percent:
            raise ValueError("minimum_n1_percent must be below maximum_n1_percent")

    def limited_command(self, commanded_n1_percent: float) -> float:
        """Return a finite command saturated to the modeled N1 envelope."""
        if not np.isfinite(commanded_n1_percent):
            raise ValueError("commanded_n1_percent must be finite")
        return float(
            np.clip(
                commanded_n1_percent,
                self.minimum_n1_percent,
                self.maximum_n1_percent,
            )
        )

    def _validated_state(self, n1_percent: float) -> float:
        if not np.isfinite(n1_percent):
            raise ValueError("n1_percent must be finite")
        if not self.minimum_n1_percent <= n1_percent <= self.maximum_n1_percent:
            raise ValueError("n1_percent is outside the modeled N1 envelope")
        return float(n1_percent)

    def derivative(self, n1_percent: float, commanded_n1_percent: float) -> float:
        """Return effective N1 rate in percentage points per second."""
        state = self._validated_state(n1_percent)
        command = self.limited_command(commanded_n1_percent)
        time_constant = (
            self.rise_time_constant_s if command >= state else self.fall_time_constant_s
        )
        return (command - state) / time_constant

    def step(
        self,
        n1_percent: float,
        commanded_n1_percent: float,
        interval_s: float,
    ) -> float:
        """Advance the state exactly for one zero-order-held command interval."""
        state = self._validated_state(n1_percent)
        command = self.limited_command(commanded_n1_percent)
        if not np.isfinite(interval_s) or interval_s < 0:
            raise ValueError("interval_s must be finite and non-negative")
        time_constant = (
            self.rise_time_constant_s if command >= state else self.fall_time_constant_s
        )
        next_state = command + (state - command) * np.exp(-interval_s / time_constant)
        return float(
            np.clip(
                next_state,
                self.minimum_n1_percent,
                self.maximum_n1_percent,
            )
        )

    def simulate_zoh(
        self,
        time_s: ArrayLike,
        commanded_n1_percent: ArrayLike,
        *,
        initial_n1_percent: float,
    ) -> FloatArray:
        """Simulate sampled commands and return N1 at the supplied sample times."""
        time = np.asarray(time_s, dtype=float)
        command = np.asarray(commanded_n1_percent, dtype=float)
        if time.ndim != 1 or command.shape != time.shape:
            raise ValueError("time and command must be equal-length vectors")
        if len(time) < 1 or np.any(~np.isfinite(time)):
            raise ValueError("time_s must contain at least one finite sample")
        if len(time) > 1 and np.any(np.diff(time) <= 0):
            raise ValueError("time_s must be strictly increasing")
        if np.any(~np.isfinite(command)):
            raise ValueError("commanded_n1_percent must be finite")

        state = np.empty_like(time)
        state[0] = self._validated_state(initial_n1_percent)
        for index in range(len(time) - 1):
            state[index + 1] = self.step(
                float(state[index]),
                float(command[index]),
                float(time[index + 1] - time[index]),
            )
        return state


def total_temperature_ratio(
    mach: ArrayLike, static_temperature_k: ArrayLike
) -> FloatArray:
    """Return inlet total-temperature ratio relative to ISA sea level."""
    mach_array = np.asarray(mach, dtype=float)
    temperature = np.asarray(static_temperature_k, dtype=float)
    if np.any(~np.isfinite(mach_array)) or np.any(mach_array < 0):
        raise ValueError("mach must be finite and non-negative")
    if np.any(~np.isfinite(temperature)) or np.any(temperature <= 0):
        raise ValueError("static_temperature_k must be finite and positive")
    return temperature / aero.T0 * (1.0 + aero.gamma1 * mach_array**2)


def fan_speed_fraction(
    n1_percent: ArrayLike,
    mach: ArrayLike,
    static_temperature_k: ArrayLike,
    *,
    corrected: bool,
) -> FloatArray:
    """Return raw or inlet-total-temperature-corrected N1 fraction."""
    n1 = np.asarray(n1_percent, dtype=float) / 100.0
    if np.any(~np.isfinite(n1)) or np.any(n1 < 0):
        raise ValueError("n1_percent must be finite and non-negative")
    if not corrected:
        return n1
    return n1 / np.sqrt(total_temperature_ratio(mach, static_temperature_k))


def thrust_map_features(
    *,
    n1_percent: ArrayLike,
    mach: ArrayLike,
    pressure_ratio: ArrayLike,
    static_temperature_k: ArrayLike,
    pressure_exponent: float,
    feature_set: str,
    corrected_fan_speed: bool,
) -> FloatArray:
    """Construct a dimensionless effective-thrust design matrix."""
    if feature_set not in THRUST_MAP_FEATURE_SETS:
        raise ValueError(f"unknown feature set: {feature_set}")
    if not np.isfinite(pressure_exponent) or pressure_exponent <= 0:
        raise ValueError("pressure_exponent must be finite and positive")
    mach_array = np.asarray(mach, dtype=float)
    delta = np.asarray(pressure_ratio, dtype=float)
    if np.any(~np.isfinite(delta)) or np.any(delta <= 0):
        raise ValueError("pressure_ratio must be finite and positive")
    fan_speed = fan_speed_fraction(
        n1_percent,
        mach_array,
        static_temperature_k,
        corrected=corrected_fan_speed,
    )
    fan_centered = fan_speed - 0.8
    mach_centered = mach_array - 0.6
    available = {
        "constant": np.ones_like(fan_centered),
        "fan_speed_centered": fan_centered,
        "fan_speed_centered_squared": fan_centered**2,
        "fan_speed_centered_cubed": fan_centered**3,
        "mach_centered": mach_centered,
        "mach_centered_squared": mach_centered**2,
        "fan_speed_mach_interaction": fan_centered * mach_centered,
    }
    base = np.column_stack(
        [available[name] for name in THRUST_MAP_FEATURE_SETS[feature_set]]
    )
    return delta[:, None] ** pressure_exponent * base


@dataclass(frozen=True)
class EffectiveThrustMap:
    """Fitted effective total-aircraft thrust map."""

    rated_total_thrust_n: float
    pressure_exponent: float
    feature_set: str
    corrected_fan_speed: bool
    coefficients: FloatArray

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = len(THRUST_MAP_FEATURE_SETS.get(self.feature_set, ()))
        if self.rated_total_thrust_n <= 0:
            raise ValueError("rated_total_thrust_n must be positive")
        if coefficients.ndim != 1 or len(coefficients) != expected:
            raise ValueError("coefficient count does not match feature set")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        object.__setattr__(self, "coefficients", coefficients)

    def raw_thrust_n(
        self,
        *,
        n1_percent: ArrayLike,
        mach: ArrayLike,
        pressure_ratio: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> FloatArray:
        design = thrust_map_features(
            n1_percent=n1_percent,
            mach=mach,
            pressure_ratio=pressure_ratio,
            static_temperature_k=static_temperature_k,
            pressure_exponent=self.pressure_exponent,
            feature_set=self.feature_set,
            corrected_fan_speed=self.corrected_fan_speed,
        )
        return self.rated_total_thrust_n * (design @ self.coefficients)

    def raw_thrust_n_scalar(
        self,
        *,
        n1_percent: float,
        mach: float,
        pressure_ratio: float,
        static_temperature_k: float,
    ) -> float:
        """Scalar fast path for numerical integration."""
        fan_speed = n1_percent / 100.0
        if self.corrected_fan_speed:
            theta_total = static_temperature_k / aero.T0 * (1.0 + aero.gamma1 * mach**2)
            fan_speed /= np.sqrt(theta_total)
        fan_centered = fan_speed - 0.8
        mach_centered = mach - 0.6
        available = {
            "constant": 1.0,
            "fan_speed_centered": fan_centered,
            "fan_speed_centered_squared": fan_centered**2,
            "fan_speed_centered_cubed": fan_centered**3,
            "mach_centered": mach_centered,
            "mach_centered_squared": mach_centered**2,
            "fan_speed_mach_interaction": fan_centered * mach_centered,
        }
        base = sum(
            coefficient * available[name]
            for coefficient, name in zip(
                self.coefficients,
                THRUST_MAP_FEATURE_SETS[self.feature_set],
                strict=True,
            )
        )
        return float(
            self.rated_total_thrust_n * pressure_ratio**self.pressure_exponent * base
        )

    def thrust_n(
        self,
        *,
        n1_percent: ArrayLike,
        mach: ArrayLike,
        pressure_ratio: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> FloatArray:
        """Predict non-negative effective thrust within the fitted envelope."""
        return np.maximum(
            0.0,
            self.raw_thrust_n(
                n1_percent=n1_percent,
                mach=mach,
                pressure_ratio=pressure_ratio,
                static_temperature_k=static_temperature_k,
            ),
        )

    def thrust_n_scalar(
        self,
        *,
        n1_percent: float,
        mach: float,
        pressure_ratio: float,
        static_temperature_k: float,
    ) -> float:
        return max(
            0.0,
            self.raw_thrust_n_scalar(
                n1_percent=n1_percent,
                mach=mach,
                pressure_ratio=pressure_ratio,
                static_temperature_k=static_temperature_k,
            ),
        )

    def n1_for_thrust_scalar(
        self,
        *,
        thrust_n: float,
        mach: float,
        pressure_ratio: float,
        static_temperature_k: float,
        minimum_n1_percent: float,
        maximum_n1_percent: float,
        tolerance_n1_percent: float = 1e-6,
    ) -> float:
        """Invert the monotone local thrust map with bounded bisection.

        Requested thrust outside the supplied N1 interval is saturated to the
        corresponding endpoint.  The method rejects a non-increasing local map
        rather than silently choosing an ambiguous engine command.
        """
        values = (
            thrust_n,
            mach,
            pressure_ratio,
            static_temperature_k,
            minimum_n1_percent,
            maximum_n1_percent,
            tolerance_n1_percent,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("thrust inversion inputs must be finite")
        if thrust_n < 0.0:
            raise ValueError("thrust_n must be non-negative")
        if minimum_n1_percent >= maximum_n1_percent:
            raise ValueError("minimum N1 must be below maximum N1")
        if tolerance_n1_percent <= 0.0:
            raise ValueError("tolerance_n1_percent must be positive")

        lower = float(minimum_n1_percent)
        upper = float(maximum_n1_percent)
        lower_thrust = self.thrust_n_scalar(
            n1_percent=lower,
            mach=mach,
            pressure_ratio=pressure_ratio,
            static_temperature_k=static_temperature_k,
        )
        upper_thrust = self.thrust_n_scalar(
            n1_percent=upper,
            mach=mach,
            pressure_ratio=pressure_ratio,
            static_temperature_k=static_temperature_k,
        )
        if upper_thrust <= lower_thrust:
            raise ValueError(
                "effective thrust map is not increasing over the N1 interval"
            )
        if thrust_n <= lower_thrust:
            return lower
        if thrust_n >= upper_thrust:
            return upper
        while upper - lower > tolerance_n1_percent:
            midpoint = 0.5 * (lower + upper)
            midpoint_thrust = self.thrust_n_scalar(
                n1_percent=midpoint,
                mach=mach,
                pressure_ratio=pressure_ratio,
                static_temperature_k=static_temperature_k,
            )
            if midpoint_thrust < thrust_n:
                lower = midpoint
            else:
                upper = midpoint
        return 0.5 * (lower + upper)


@dataclass(frozen=True)
class MeasuredN1ThrustAdapter:
    """Convert measured N1 and ambient state to replay net-thrust input.

    The wrapped map is conditional on its drag prior and QAR force-balance
    target. This adapter names that boundary explicitly; it is not a
    throttle-to-thrust or independent engine-deck model.
    """

    thrust_map: EffectiveThrustMap

    def net_thrust_n(
        self,
        *,
        n1_percent: ArrayLike,
        mach: ArrayLike,
        pressure_ratio: ArrayLike,
        static_temperature_k: ArrayLike,
    ) -> FloatArray:
        """Return non-negative effective total-aircraft thrust for replay."""
        return self.thrust_map.thrust_n(
            n1_percent=n1_percent,
            mach=mach,
            pressure_ratio=pressure_ratio,
            static_temperature_k=static_temperature_k,
        )

    def net_thrust_n_scalar(
        self,
        *,
        n1_percent: float,
        mach: float,
        pressure_ratio: float,
        static_temperature_k: float,
    ) -> float:
        """Scalar replay-adapter path."""
        return self.thrust_map.thrust_n_scalar(
            n1_percent=n1_percent,
            mach=mach,
            pressure_ratio=pressure_ratio,
            static_temperature_k=static_temperature_k,
        )
