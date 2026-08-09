"""Frame-explicit kinematic trajectory replay on the WGS 84 ellipsoid."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]

WGS84_A_M = 6_378_137.0
WGS84_INV_F = 298.257_223_563
WGS84_F = 1.0 / WGS84_INV_F
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_MEAN_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class KinematicTrajectory:
    """Position history and the NED ground velocity used to integrate it."""

    latitude_rad: FloatArray
    longitude_rad: FloatArray
    altitude_m: FloatArray
    north_velocity_mps: FloatArray
    east_velocity_mps: FloatArray
    down_velocity_mps: FloatArray


@dataclass(frozen=True)
class ReplayMetrics:
    """Open-loop position-error summary against an observed trajectory."""

    horizontal_rmse_m: float
    horizontal_p95_m: float
    horizontal_final_m: float
    vertical_rmse_m: float
    vertical_p95_m: float
    vertical_final_m: float


def _arrays(*values: NDArray | list[float]) -> list[FloatArray]:
    arrays = [np.asarray(value, dtype=float) for value in values]
    if not arrays or any(array.ndim != 1 for array in arrays):
        raise ValueError("all inputs must be one-dimensional")
    if len({len(array) for array in arrays}) != 1:
        raise ValueError("all inputs must have equal lengths")
    return arrays


def air_velocity_from_orientation(
    true_airspeed_mps: FloatArray,
    angle_of_attack_rad: FloatArray,
    sideslip_rad: FloatArray,
    roll_rad: FloatArray,
    pitch_rad: FloatArray,
    heading_rad: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Rotate body-axis air velocity into north-east-down coordinates.

    Body axes are x forward, y right, z down. Euler angles use the aerospace
    3-2-1 convention: roll right-wing-down, pitch nose-up, and heading clockwise
    from true north.
    """
    tas, alpha, beta, roll, pitch, heading = _arrays(
        true_airspeed_mps,
        angle_of_attack_rad,
        sideslip_rad,
        roll_rad,
        pitch_rad,
        heading_rad,
    )
    ca, sa = np.cos(alpha), np.sin(alpha)
    cb, sb = np.cos(beta), np.sin(beta)
    body_x = tas * ca * cb
    body_y = tas * sb
    body_z = tas * sa * cb

    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    ch, sh = np.cos(heading), np.sin(heading)
    north = (
        cp * ch * body_x
        + (sr * sp * ch - cr * sh) * body_y
        + (cr * sp * ch + sr * sh) * body_z
    )
    east = (
        cp * sh * body_x
        + (sr * sp * sh + cr * ch) * body_y
        + (cr * sp * sh - sr * ch) * body_z
    )
    down = -sp * body_x + sr * cp * body_y + cr * cp * body_z
    return north, east, down


def air_velocity_from_heading_flight_path(
    true_airspeed_mps: FloatArray,
    heading_rad: FloatArray,
    flight_path_angle_rad: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build NED air velocity from true heading and upward-positive path angle."""
    tas, heading, gamma = _arrays(true_airspeed_mps, heading_rad, flight_path_angle_rad)
    horizontal = tas * np.cos(gamma)
    return (
        horizontal * np.cos(heading),
        horizontal * np.sin(heading),
        -tas * np.sin(gamma),
    )


def ground_velocity_from_track(
    groundspeed_mps: FloatArray,
    track_rad: FloatArray,
    vertical_rate_mps: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Build NED ground velocity from track and upward-positive vertical rate."""
    speed, track, vertical_rate = _arrays(groundspeed_mps, track_rad, vertical_rate_mps)
    return (
        speed * np.cos(track),
        speed * np.sin(track),
        -vertical_rate,
    )


def _curvature_radii(latitude_rad: float) -> tuple[float, float]:
    sin_lat = np.sin(latitude_rad)
    denominator = np.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    prime_vertical = WGS84_A_M / denominator
    meridian = WGS84_A_M * (1.0 - WGS84_E2) / denominator**3
    return float(meridian), float(prime_vertical)


def integrate_wgs84(
    time_s: FloatArray,
    initial_latitude_rad: float,
    initial_longitude_rad: float,
    initial_altitude_m: float,
    north_velocity_mps: FloatArray,
    east_velocity_mps: FloatArray,
    down_velocity_mps: FloatArray,
) -> KinematicTrajectory:
    """Integrate NED velocity on WGS 84 using trapezoidal one-step updates."""
    time, north, east, down = _arrays(
        time_s, north_velocity_mps, east_velocity_mps, down_velocity_mps
    )
    if len(time) == 0:
        raise ValueError("at least one sample is required")
    if np.any(~np.isfinite(np.column_stack((time, north, east, down)))):
        raise ValueError("time and velocity inputs must be finite")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")

    latitude = np.empty(len(time), dtype=float)
    longitude = np.empty(len(time), dtype=float)
    altitude = np.empty(len(time), dtype=float)
    latitude[0] = initial_latitude_rad
    longitude[0] = initial_longitude_rad
    altitude[0] = initial_altitude_m
    for index in range(1, len(time)):
        step = dt[index - 1]
        vn = 0.5 * (north[index - 1] + north[index])
        ve = 0.5 * (east[index - 1] + east[index])
        vd = 0.5 * (down[index - 1] + down[index])
        midpoint_altitude = altitude[index - 1] - 0.5 * vd * step
        meridian, prime_vertical = _curvature_radii(latitude[index - 1])
        latitude[index] = latitude[index - 1] + (
            vn * step / (meridian + midpoint_altitude)
        )
        cos_latitude = np.cos(0.5 * (latitude[index - 1] + latitude[index]))
        longitude[index] = longitude[index - 1] + (
            ve * step / ((prime_vertical + midpoint_altitude) * cos_latitude)
        )
        altitude[index] = altitude[index - 1] - vd * step
    return KinematicTrajectory(latitude, longitude, altitude, north, east, down)


def differentiate_wgs84_intervals(
    time_s: FloatArray,
    latitude_rad: FloatArray,
    longitude_rad: FloatArray,
    altitude_m: FloatArray,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Recover mean NED velocity over each observed geodetic interval."""
    time, latitude, longitude, altitude = _arrays(
        time_s, latitude_rad, longitude_rad, altitude_m
    )
    if len(time) < 2:
        raise ValueError("at least two position samples are required")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")
    north = np.empty(len(time) - 1, dtype=float)
    east = np.empty(len(time) - 1, dtype=float)
    down = -np.diff(altitude) / dt
    for index in range(len(north)):
        midpoint_altitude = 0.5 * (altitude[index] + altitude[index + 1])
        meridian, prime_vertical = _curvature_radii(latitude[index])
        north[index] = (
            (latitude[index + 1] - latitude[index])
            * (meridian + midpoint_altitude)
            / dt[index]
        )
        midpoint_latitude = 0.5 * (latitude[index] + latitude[index + 1])
        east[index] = (
            (longitude[index + 1] - longitude[index])
            * (prime_vertical + midpoint_altitude)
            * np.cos(midpoint_latitude)
            / dt[index]
        )
    return north, east, down


def integrate_wgs84_intervals(
    time_s: FloatArray,
    initial_latitude_rad: float,
    initial_longitude_rad: float,
    initial_altitude_m: float,
    north_interval_velocity_mps: FloatArray,
    east_interval_velocity_mps: FloatArray,
    down_interval_velocity_mps: FloatArray,
) -> KinematicTrajectory:
    """Integrate one mean NED velocity per time interval on WGS 84."""
    time = np.asarray(time_s, dtype=float)
    north, east, down = _arrays(
        north_interval_velocity_mps,
        east_interval_velocity_mps,
        down_interval_velocity_mps,
    )
    if len(time) != len(north) + 1:
        raise ValueError("interval velocity arrays must have len(time_s) - 1 samples")
    dt = np.diff(time)
    if np.any(dt <= 0):
        raise ValueError("time_s must be strictly increasing")

    latitude = np.empty(len(time), dtype=float)
    longitude = np.empty(len(time), dtype=float)
    altitude = np.empty(len(time), dtype=float)
    latitude[0] = initial_latitude_rad
    longitude[0] = initial_longitude_rad
    altitude[0] = initial_altitude_m
    for index, step in enumerate(dt, start=1):
        vn = north[index - 1]
        ve = east[index - 1]
        vd = down[index - 1]
        midpoint_altitude = altitude[index - 1] - 0.5 * vd * step
        meridian, prime_vertical = _curvature_radii(latitude[index - 1])
        latitude[index] = latitude[index - 1] + (
            vn * step / (meridian + midpoint_altitude)
        )
        cos_latitude = np.cos(0.5 * (latitude[index - 1] + latitude[index]))
        longitude[index] = longitude[index - 1] + (
            ve * step / ((prime_vertical + midpoint_altitude) * cos_latitude)
        )
        altitude[index] = altitude[index - 1] - vd * step

    # A trajectory stores node-aligned diagnostics. Repeat the final interval at
    # the final node; position integration itself uses only interval values.
    node_north = np.append(north, north[-1])
    node_east = np.append(east, east[-1])
    node_down = np.append(down, down[-1])
    return KinematicTrajectory(
        latitude, longitude, altitude, node_north, node_east, node_down
    )


def horizontal_distance_m(
    latitude_a_rad: FloatArray,
    longitude_a_rad: FloatArray,
    latitude_b_rad: FloatArray,
    longitude_b_rad: FloatArray,
) -> FloatArray:
    """Great-circle horizontal separation using the IUGG mean Earth radius."""
    lat_a, lon_a, lat_b, lon_b = _arrays(
        latitude_a_rad, longitude_a_rad, latitude_b_rad, longitude_b_rad
    )
    dlat = lat_b - lat_a
    dlon = lon_b - lon_a
    haversine = np.sin(dlat / 2.0) ** 2 + (
        np.cos(lat_a) * np.cos(lat_b) * np.sin(dlon / 2.0) ** 2
    )
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(haversine, 0.0, 1.0)))
    return EARTH_MEAN_RADIUS_M * central_angle


def replay_metrics(
    predicted: KinematicTrajectory,
    observed_latitude_rad: FloatArray,
    observed_longitude_rad: FloatArray,
    observed_altitude_m: FloatArray,
) -> ReplayMetrics:
    """Calculate complete-flight open-loop horizontal and vertical errors."""
    observed_lat, observed_lon, observed_alt = _arrays(
        observed_latitude_rad, observed_longitude_rad, observed_altitude_m
    )
    if len(observed_lat) != len(predicted.latitude_rad):
        raise ValueError("predicted and observed trajectories must have equal lengths")
    horizontal = horizontal_distance_m(
        predicted.latitude_rad,
        predicted.longitude_rad,
        observed_lat,
        observed_lon,
    )
    vertical = predicted.altitude_m - observed_alt
    return ReplayMetrics(
        horizontal_rmse_m=float(np.sqrt(np.mean(horizontal**2))),
        horizontal_p95_m=float(np.quantile(horizontal, 0.95)),
        horizontal_final_m=float(horizontal[-1]),
        vertical_rmse_m=float(np.sqrt(np.mean(vertical**2))),
        vertical_p95_m=float(np.quantile(np.abs(vertical), 0.95)),
        vertical_final_m=float(vertical[-1]),
    )
