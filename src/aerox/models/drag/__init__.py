"""Aircraft drag models.

Build the drag model of an aircraft with :func:`build_drag_model` and evaluate
it with ``model.drag_coefficient(cl, mach, reynolds_per_m)``.

The component equations, the wave-drag model, and the published comparison
models are available from the submodules ``components``, ``wave``,
``geometry``, ``builder``, and ``reference``.
"""

from .builder import (
    DragBuildReport,
    build_drag_model,
    build_drag_report,
    build_reference_drag_polar,
)
from .core import DragCondition, DragModel
from .geometry import GeometryDragModel, TransonicGeometryDragModel
from .polar import ParabolicDragPolar

__all__ = [
    "DragBuildReport",
    "DragCondition",
    "DragModel",
    "GeometryDragModel",
    "ParabolicDragPolar",
    "TransonicGeometryDragModel",
    "build_drag_model",
    "build_drag_report",
    "build_reference_drag_polar",
]
