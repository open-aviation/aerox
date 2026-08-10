"""Aerox flight-management trajectory core."""

from aerox.fms.planning.econ_cruise import (
    EconomyCruiseOptimizer,
    EconomyCruisePoint,
)
from aerox.fms.environment import EnvironmentSample, SpatialEnvironmentProfile
from aerox.fms.intent import FlightIntent, Waypoint
from aerox.fms.performance import FuelFlowModel, LocalPerformanceLimits, ProfilePerformance
from aerox.fms.planning.advisory import plan_advisory_energy_profile
from aerox.fms.planning.energy import plan_energy_shaped_profile
from aerox.fms.planning.feasibility import (
    ProfileFeasibilityResult,
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
from aerox.fms.planning.speed import (
    ContinuousSpeedMassPlan,
    CruiseSpeedStrategy,
    plan_continuous_speed_mass_profile,
)
from aerox.fms.route import (
    RouteProjection,
    SpatialTrajectory,
    SpatialTrajectoryGrid,
    SpatialTrajectorySample,
    inherit_state_profile,
)
from aerox.fms.planning.vertical import (
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
    "CASMachClimbSchedule",
    "CASMachDescentSchedule",
    "ContinuousSpeedMassPlan",
    "CruiseSpeedStrategy",
    "EnvironmentSample",
    "EconomyCruiseOptimizer",
    "EconomyCruisePoint",
    "FlightIntent",
    "FuelFlowModel",
    "LocalPerformanceLimits",
    "ModeConditionedSpeedIntent",
    "ProfilePerformance",
    "ProfileFeasibilityResult",
    "RouteProjection",
    "SpatialEnvironmentProfile",
    "SpatialTrajectory",
    "SpatialTrajectoryGrid",
    "SpatialTrajectorySample",
    "SpeedIntentPrediction",
    "TopOfClimbPlan",
    "TopOfDescentPlan",
    "Waypoint",
    "inherit_state_profile",
    "plan_advisory_energy_profile",
    "plan_continuous_speed_mass_profile",
    "plan_energy_shaped_profile",
    "plan_feasible_profile",
    "plan_top_of_climb",
    "plan_top_of_descent",
    "speed_intent_design",
    "vertical_phase",
]
