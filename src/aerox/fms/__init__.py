"""Aerox flight-management trajectory core."""

from aerox.fms.econ_cruise import (
    CruiseFuelFlowModel,
    EconomyCruiseOptimizer,
    EconomyCruisePoint,
)
from aerox.fms.environment import EnvironmentSample, SpatialEnvironmentProfile
from aerox.fms.guidance import (
    FMSTrajectoryResult,
    GuidanceTarget,
    SpatialPathGuidance,
    simulate_fms_trajectory_zoh,
)
from aerox.fms.intent import FlightIntent, Waypoint
from aerox.fms.profile import (
    PointMassProfilePerformance,
    ConstantMaximumThrust,
    LocalPerformanceLimits,
    PressureScaledMinimumThrust,
    ProfileFeasibilityResult,
    QARDemonstratedClimbThrustCapacity,
    QAREffectiveClimbThrustCeiling,
    plan_advisory_energy_profile,
    plan_energy_shaped_profile,
    plan_feasible_profile,
)
from aerox.fms.speed_intent import (
    AFFINE_SELECTOR_FEATURES,
    PHASE_CONSTANT_FEATURES,
    ModeConditionedSpeedIntent,
    SpeedIntentPrediction,
    speed_intent_design,
    vertical_phase,
)
from aerox.fms.speed_profile import (
    ContinuousSpeedMassPlan,
    CruiseSpeedStrategy,
    FuelFlowModel,
    plan_continuous_speed_mass_profile,
)
from aerox.fms.trajectory import (
    RouteProjection,
    SpatialTrajectory,
    SpatialTrajectorySample,
    ground_track_to_air_heading,
    inherit_state_profile,
)
from aerox.fms.vertical import (
    CASMachClimbSchedule,
    CASMachDescentSchedule,
    TopOfClimbPlan,
    TopOfDescentPlan,
    plan_top_of_climb,
    plan_top_of_descent,
)

__all__ = [
    "AFFINE_SELECTOR_FEATURES",
    "PHASE_CONSTANT_FEATURES",
    "PointMassProfilePerformance",
    "CASMachClimbSchedule",
    "CASMachDescentSchedule",
    "ConstantMaximumThrust",
    "ContinuousSpeedMassPlan",
    "CruiseFuelFlowModel",
    "CruiseSpeedStrategy",
    "EnvironmentSample",
    "EconomyCruiseOptimizer",
    "EconomyCruisePoint",
    "FMSTrajectoryResult",
    "FlightIntent",
    "FuelFlowModel",
    "GuidanceTarget",
    "LocalPerformanceLimits",
    "ModeConditionedSpeedIntent",
    "PressureScaledMinimumThrust",
    "ProfileFeasibilityResult",
    "QARDemonstratedClimbThrustCapacity",
    "QAREffectiveClimbThrustCeiling",
    "RouteProjection",
    "SpatialEnvironmentProfile",
    "SpatialPathGuidance",
    "SpatialTrajectory",
    "SpatialTrajectorySample",
    "SpeedIntentPrediction",
    "TopOfClimbPlan",
    "TopOfDescentPlan",
    "Waypoint",
    "ground_track_to_air_heading",
    "inherit_state_profile",
    "plan_advisory_energy_profile",
    "plan_continuous_speed_mass_profile",
    "plan_energy_shaped_profile",
    "plan_feasible_profile",
    "plan_top_of_climb",
    "plan_top_of_descent",
    "simulate_fms_trajectory_zoh",
    "speed_intent_design",
    "vertical_phase",
]
