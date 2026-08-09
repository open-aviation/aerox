"""Condition-aware aircraft drag modeling.

The package separates the runtime model contract, component and geometry
equations, aircraft-data construction, low-order polar projections, and
published comparison formulations.
"""

from .builder import (
    GEOMETRY_DRAG_MODEL_ID,
    KORN_LOCK_WAVE_DRAG_MODEL_ID,
    DragBuildReport,
    GeometryScalingDefaults,
    ReferenceDragCondition,
    ResolvedDragGeometry,
    build_drag_model,
    build_drag_report,
    build_reference_drag_polar,
    derive_reference_drag_condition,
    resolve_drag_geometry,
)
from .components import (
    ParasiteDragComponent,
    fuselage_form_factor,
    fuselage_wetted_area,
    laminar_flat_plate_skin_friction,
    lifting_surface_form_factor,
    lifting_surface_wetted_area,
    turbulent_flat_plate_skin_friction,
)
from .core import DragBreakdown, DragCondition, DragModel
from .geometry import (
    GeometryDragBreakdown,
    GeometryDragModel,
    TransonicGeometryDragBreakdown,
    TransonicGeometryDragModel,
    geometry_drag_model_from_aircraft,
    geometry_drag_model_from_dimensions,
)
from .polar import ParabolicDragPolar, ParabolicWaveDragModel, project_parabolic_polar
from .reference import (
    HyperbolicTangentWaveDrag,
    MachDependentInducedDrag,
    PollSchumannDragModel,
    TransonicCompressibilityDragModel,
)
from .wave import KornLockWaveDrag

__all__ = [
    "GEOMETRY_DRAG_MODEL_ID",
    "KORN_LOCK_WAVE_DRAG_MODEL_ID",
    "DragBreakdown",
    "DragBuildReport",
    "DragCondition",
    "DragModel",
    "GeometryDragBreakdown",
    "GeometryDragModel",
    "GeometryScalingDefaults",
    "HyperbolicTangentWaveDrag",
    "KornLockWaveDrag",
    "MachDependentInducedDrag",
    "ParabolicDragPolar",
    "ParabolicWaveDragModel",
    "ParasiteDragComponent",
    "PollSchumannDragModel",
    "ReferenceDragCondition",
    "ResolvedDragGeometry",
    "TransonicCompressibilityDragModel",
    "TransonicGeometryDragBreakdown",
    "TransonicGeometryDragModel",
    "build_drag_model",
    "build_drag_report",
    "build_reference_drag_polar",
    "derive_reference_drag_condition",
    "fuselage_form_factor",
    "fuselage_wetted_area",
    "geometry_drag_model_from_aircraft",
    "geometry_drag_model_from_dimensions",
    "laminar_flat_plate_skin_friction",
    "lifting_surface_form_factor",
    "lifting_surface_wetted_area",
    "project_parabolic_polar",
    "resolve_drag_geometry",
    "turbulent_flat_plate_skin_friction",
]
