"""Observable guidance regimes from trajectory and engine signals.

The estimator reports behavioural evidence rather than decoded Flight Mode
Annunciator labels. It is intended for QAR datasets in which the FMA words are
absent: concurrent vertical, speed, thrust, and lateral regimes are inferred
independently, then combined into an explicitly ambiguous guidance family.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.signal import savgol_filter


FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]


@dataclass(frozen=True)
class GuidanceModeThresholds:
    """Predeclared thresholds for 1 Hz effective-mode segmentation."""

    derivative_window_s: int = 61
    variability_window_s: int = 91
    minimum_dwell_s: int = 15
    level_vertical_rate_ft_min: float = 250.0
    climb_vertical_rate_ft_min: float = 400.0
    descent_vertical_rate_ft_min: float = -400.0
    altitude_hold_error_ft: float = 400.0
    altitude_hold_variability_ft: float = 100.0
    altitude_capture_error_ft: float = 1_500.0
    altitude_capture_maximum_rate_ft_min: float = 1_500.0
    cas_hold_std_kts: float = 3.0
    mach_hold_std: float = 0.006
    speed_change_rate_kts_s: float = 0.04
    mach_preferred_altitude_ft: float = 25_000.0
    selected_cas_tolerance_kts: float = 5.0
    selected_mach_tolerance: float = 0.01
    climb_n1_margin_percent: float = 3.0
    idle_n1_threshold_percent: float = 45.0
    turn_rate_threshold_deg_s: float = 0.25
    bank_turn_threshold_deg: float = 5.0
    selected_heading_tolerance_deg: float = 4.0

    def __post_init__(self) -> None:
        if self.derivative_window_s < 5 or self.derivative_window_s % 2 == 0:
            raise ValueError("derivative_window_s must be odd and at least 5")
        if self.variability_window_s < 5 or self.variability_window_s % 2 == 0:
            raise ValueError("variability_window_s must be odd and at least 5")
        if self.minimum_dwell_s < 1:
            raise ValueError("minimum_dwell_s must be positive")


@dataclass(frozen=True)
class InferredGuidanceModes:
    """Concurrent effective-mode estimates and diagnostic features."""

    vertical_mode: StringArray
    speed_mode: StringArray
    thrust_mode: StringArray
    lateral_mode: StringArray
    speed_target_evidence: StringArray
    guidance_family: StringArray
    vertical_rate_ft_min: FloatArray
    cas_rate_kts_s: FloatArray
    cas_rolling_std_kts: FloatArray
    mach_rolling_std: FloatArray


def _finite_interpolate(values: ArrayLike) -> FloatArray:
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("mode-estimator inputs must be one-dimensional")
    finite = np.isfinite(data)
    if not np.any(finite):
        return np.zeros_like(data)
    indices = np.arange(len(data), dtype=float)
    return np.interp(indices, indices[finite], data[finite])


def _smooth_derivative(
    values: ArrayLike, window: int, scale: float = 1.0
) -> FloatArray:
    data = _finite_interpolate(values)
    if len(data) < 5:
        return np.gradient(data) * scale
    usable = min(window, len(data) if len(data) % 2 else len(data) - 1)
    usable = max(5, usable)
    return savgol_filter(data, usable, 2, deriv=1, delta=1.0, mode="interp") * scale


def _rolling_std(values: ArrayLike, window: int) -> FloatArray:
    data = _finite_interpolate(values)
    if len(data) == 0:
        return data
    width = min(window, len(data))
    kernel = np.ones(width, dtype=float) / width
    mean = np.convolve(data, kernel, mode="same")
    mean_square = np.convolve(data**2, kernel, mode="same")
    result = np.sqrt(np.maximum(0.0, mean_square - mean**2))
    edge = width // 2
    if edge > 0 and len(data) > edge:
        result[:edge] = result[edge]
        result[-edge:] = result[-edge - 1]
    return result


def _circular_difference_deg(left: ArrayLike, right: ArrayLike) -> FloatArray:
    return (
        np.asarray(left, dtype=float) - np.asarray(right, dtype=float) + 180.0
    ) % 360.0 - 180.0


def _persistent(labels: StringArray, minimum_dwell: int) -> StringArray:
    """Suppress short label flicker while retaining explicit uncertainty."""
    result = np.asarray(labels, dtype="U40").copy()
    if len(result) == 0:
        return result
    for _ in range(2):
        boundaries = np.r_[
            0, np.flatnonzero(result[1:] != result[:-1]) + 1, len(result)
        ]
        updated = result.copy()
        for start, stop in zip(boundaries[:-1], boundaries[1:], strict=True):
            if stop - start >= minimum_dwell:
                continue
            left = result[start - 1] if start > 0 else None
            right = result[stop] if stop < len(result) else None
            replacement = left if left is not None and left == right else "UNRESOLVED"
            updated[start:stop] = replacement
        result = updated
    return result


def infer_guidance_modes(
    *,
    pressure_altitude_ft: ArrayLike,
    calibrated_airspeed_kts: ArrayLike,
    mach: ArrayLike,
    selected_altitude_ft: ArrayLike,
    selected_speed_kts: ArrayLike,
    selected_mach: ArrayLike,
    mean_n1_percent: ArrayLike,
    climb_n1_limit_percent: ArrayLike,
    heading_deg: ArrayLike,
    track_deg: ArrayLike,
    selected_heading_deg: ArrayLike,
    selected_track_deg: ArrayLike,
    bank_angle_deg: ArrayLike,
    thresholds: GuidanceModeThresholds = GuidanceModeThresholds(),
) -> InferredGuidanceModes:
    """Infer persistent effective guidance regimes from aligned 1 Hz signals."""
    altitude = np.asarray(pressure_altitude_ft, dtype=float)
    cas = np.asarray(calibrated_airspeed_kts, dtype=float)
    mach_array = np.asarray(mach, dtype=float)
    arrays = [
        altitude,
        cas,
        mach_array,
        np.asarray(selected_altitude_ft),
        np.asarray(selected_speed_kts),
        np.asarray(selected_mach),
        np.asarray(mean_n1_percent),
        np.asarray(climb_n1_limit_percent),
        np.asarray(heading_deg),
        np.asarray(track_deg),
        np.asarray(selected_heading_deg),
        np.asarray(selected_track_deg),
        np.asarray(bank_angle_deg),
    ]
    if altitude.ndim != 1 or any(array.shape != altitude.shape for array in arrays):
        raise ValueError(
            "all mode-estimator inputs must be aligned one-dimensional arrays"
        )

    selected_altitude = np.asarray(selected_altitude_ft, dtype=float)
    selected_speed = np.asarray(selected_speed_kts, dtype=float)
    selected_mach_array = np.asarray(selected_mach, dtype=float)
    n1 = np.asarray(mean_n1_percent, dtype=float)
    climb_limit = np.asarray(climb_n1_limit_percent, dtype=float)
    heading = np.asarray(heading_deg, dtype=float) % 360.0
    track = np.asarray(track_deg, dtype=float) % 360.0
    heading_selected = np.asarray(selected_heading_deg, dtype=float) % 360.0
    track_selected = np.asarray(selected_track_deg, dtype=float) % 360.0
    bank = np.asarray(bank_angle_deg, dtype=float)

    vertical_rate = _smooth_derivative(
        altitude, thresholds.derivative_window_s, scale=60.0
    )
    cas_rate = _smooth_derivative(cas, thresholds.derivative_window_s)
    heading_rate = np.rad2deg(
        _smooth_derivative(
            np.unwrap(np.deg2rad(_finite_interpolate(heading))),
            thresholds.derivative_window_s,
        )
    )
    cas_std = _rolling_std(cas, thresholds.variability_window_s)
    mach_std = _rolling_std(mach_array, thresholds.variability_window_s)
    altitude_std = _rolling_std(altitude, thresholds.variability_window_s)

    altitude_error = selected_altitude - altitude
    moving_toward_selected = altitude_error * vertical_rate > 0.0
    vertical = np.full(len(altitude), "TRANSITION", dtype="U40")
    vertical[vertical_rate >= thresholds.climb_vertical_rate_ft_min] = "CLIMB"
    vertical[vertical_rate <= thresholds.descent_vertical_rate_ft_min] = "DESCENT"
    level = np.abs(vertical_rate) <= thresholds.level_vertical_rate_ft_min
    vertical[level] = "LEVEL"
    altitude_hold = (
        level
        & (np.abs(altitude_error) <= thresholds.altitude_hold_error_ft)
        & (altitude_std <= thresholds.altitude_hold_variability_ft)
    )
    vertical[altitude_hold] = "ALT_HOLD_LIKE"
    altitude_capture = (
        moving_toward_selected
        & (np.abs(altitude_error) <= thresholds.altitude_capture_error_ft)
        & (np.abs(vertical_rate) > thresholds.level_vertical_rate_ft_min)
        & (np.abs(vertical_rate) <= thresholds.altitude_capture_maximum_rate_ft_min)
    )
    vertical[altitude_capture] = "ALT_CAPTURE_LIKE"
    vertical = _persistent(vertical, thresholds.minimum_dwell_s)

    speed = np.full(len(altitude), "VARIABLE_SPEED", dtype="U40")
    accelerating = cas_rate >= thresholds.speed_change_rate_kts_s
    decelerating = cas_rate <= -thresholds.speed_change_rate_kts_s
    mach_hold = (
        (altitude >= thresholds.mach_preferred_altitude_ft)
        & (mach_std <= thresholds.mach_hold_std)
        & ~accelerating
        & ~decelerating
    )
    cas_hold = (
        (cas_std <= thresholds.cas_hold_std_kts)
        & ~mach_hold
        & ~accelerating
        & ~decelerating
    )
    speed[cas_hold] = "CAS_HOLD_LIKE"
    speed[mach_hold] = "MACH_HOLD_LIKE"
    speed[accelerating] = "ACCELERATING"
    speed[decelerating] = "DECELERATING"
    speed = _persistent(speed, thresholds.minimum_dwell_s)

    thrust = np.full(len(altitude), "MODULATED_THRUST", dtype="U40")
    climb_like = (
        (vertical == "CLIMB")
        & np.isfinite(climb_limit)
        & (n1 >= climb_limit - thresholds.climb_n1_margin_percent)
    )
    idle_like = (vertical == "DESCENT") & (n1 <= thresholds.idle_n1_threshold_percent)
    thrust[climb_like] = "CLB_THRUST_LIKE"
    thrust[idle_like] = "IDLE_THRUST_LIKE"
    thrust = _persistent(thrust, thresholds.minimum_dwell_s)

    target = np.full(len(altitude), "TARGET_UNRESOLVED", dtype="U40")
    selected_cas_consistent = (
        (speed == "CAS_HOLD_LIKE")
        & np.isfinite(selected_speed)
        & (np.abs(cas - selected_speed) <= thresholds.selected_cas_tolerance_kts)
    )
    selected_mach_consistent = (
        (speed == "MACH_HOLD_LIKE")
        & np.isfinite(selected_mach_array)
        & (
            np.abs(mach_array - selected_mach_array)
            <= thresholds.selected_mach_tolerance
        )
    )
    target[selected_cas_consistent] = "SELECTED_CAS_CONSISTENT"
    target[selected_mach_consistent] = "SELECTED_MACH_CONSISTENT"
    target = _persistent(target, thresholds.minimum_dwell_s)

    lateral = np.full(len(altitude), "STRAIGHT_UNRESOLVED", dtype="U40")
    turning = (np.abs(heading_rate) >= thresholds.turn_rate_threshold_deg_s) | (
        np.abs(bank) >= thresholds.bank_turn_threshold_deg
    )
    selected_heading_consistent = (
        ~turning
        & np.isfinite(heading_selected)
        & (
            np.abs(_circular_difference_deg(heading, heading_selected))
            <= thresholds.selected_heading_tolerance_deg
        )
    )
    selected_track_available = np.isfinite(track_selected) & ~np.isclose(
        track_selected, 0.0
    )
    selected_track_consistent = (
        ~turning
        & selected_track_available
        & (
            np.abs(_circular_difference_deg(track, track_selected))
            <= thresholds.selected_heading_tolerance_deg
        )
    )
    lateral[selected_heading_consistent] = "HDG_SELECTED_CONSISTENT"
    lateral[selected_track_consistent] = "TRK_SELECTED_CONSISTENT"
    lateral[turning] = "TURNING"
    lateral = _persistent(lateral, thresholds.minimum_dwell_s)

    family = np.full(len(altitude), "MODE_FAMILY_UNRESOLVED", dtype="U40")
    family[vertical == "ALT_HOLD_LIKE"] = "ALT_HOLD_COMPATIBLE"
    family[vertical == "ALT_CAPTURE_LIKE"] = "ALT_CAPTURE_COMPATIBLE"
    family[(vertical == "CLIMB") & (thrust == "CLB_THRUST_LIKE")] = "CLB_OR_OP_CLB"
    family[(vertical == "DESCENT") & (thrust == "IDLE_THRUST_LIKE")] = "DES_OR_OP_DES"
    family = _persistent(family, thresholds.minimum_dwell_s)

    return InferredGuidanceModes(
        vertical_mode=vertical,
        speed_mode=speed,
        thrust_mode=thrust,
        lateral_mode=lateral,
        speed_target_evidence=target,
        guidance_family=family,
        vertical_rate_ft_min=vertical_rate,
        cas_rate_kts_s=cas_rate,
        cas_rolling_std_kts=cas_std,
        mach_rolling_std=mach_std,
    )
