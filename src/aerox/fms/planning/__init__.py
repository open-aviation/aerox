"""Aircraft-aware vertical, speed, and energy planning algorithms."""

from .advisory import plan_advisory_energy_profile
from .econ_cruise import (
    EconomyCruiseOptimizer,
    EconomyCruisePoint,
)
from .energy import plan_energy_shaped_profile
from .feasibility import ProfileFeasibilityResult, plan_feasible_profile
from .speed import (
    ContinuousSpeedMassPlan,
    CruiseSpeedStrategy,
    plan_continuous_speed_mass_profile,
)
from .vertical import (
    CASMachClimbSchedule,
    CASMachDescentSchedule,
    TopOfClimbPlan,
    TopOfDescentPlan,
    plan_top_of_climb,
    plan_top_of_descent,
)

__all__ = [
    "CASMachClimbSchedule",
    "CASMachDescentSchedule",
    "ContinuousSpeedMassPlan",
    "CruiseSpeedStrategy",
    "EconomyCruiseOptimizer",
    "EconomyCruisePoint",
    "ProfileFeasibilityResult",
    "TopOfClimbPlan",
    "TopOfDescentPlan",
    "plan_advisory_energy_profile",
    "plan_continuous_speed_mass_profile",
    "plan_energy_shaped_profile",
    "plan_feasible_profile",
    "plan_top_of_climb",
    "plan_top_of_descent",
]
