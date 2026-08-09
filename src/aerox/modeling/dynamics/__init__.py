"""Flight-dynamics equations, numerical simulation, and kinematic replay."""

from .kinematics import (
    KinematicTrajectory,
    ReplayMetrics,
    air_velocity_from_heading_flight_path,
    air_velocity_from_orientation,
    differentiate_wgs84_intervals,
    ground_velocity_from_track,
    horizontal_distance_m,
    integrate_wgs84,
    integrate_wgs84_intervals,
    replay_metrics,
)
from .point_mass import (
    HeadingRateReplayResult,
    PitchAwareLiftModel,
    PitchBankThrustInput,
    PointMassForces,
    PointMassModel,
    PointMassState,
    PointMassTrajectory,
)
from .replay import simulate_heading_rate_replay_zoh
from .simulation import simulate_point_mass_zoh

__all__ = [
    "HeadingRateReplayResult",
    "KinematicTrajectory",
    "PitchAwareLiftModel",
    "PitchBankThrustInput",
    "PointMassForces",
    "PointMassModel",
    "PointMassState",
    "PointMassTrajectory",
    "ReplayMetrics",
    "air_velocity_from_heading_flight_path",
    "air_velocity_from_orientation",
    "differentiate_wgs84_intervals",
    "ground_velocity_from_track",
    "horizontal_distance_m",
    "integrate_wgs84",
    "integrate_wgs84_intervals",
    "replay_metrics",
    "simulate_heading_rate_replay_zoh",
    "simulate_point_mass_zoh",
]
