"""Aircraft drag models.

Build the drag model of an aircraft with :func:`build_drag_model` and evaluate
it with ``model.cd(cl=..., mach=..., reynolds_per_m=...)``. For a constant
two-term polar, ``CD = cd0 + k CL²``, use :func:`build_parabolic_polar`.

The component equations, the wave-drag model, and the published comparison
models are available from the submodules ``components``, ``wave``,
``geometry``, ``builder``, and ``reference``.
"""

from .builder import (
    DragBuildReport,
    build_drag_model,
    build_drag_report,
    build_parabolic_polar,
)
from .core import DragCondition, DragModel
from .expression import DragExpression
from .geometry import GeometryDragModel, TransonicGeometryDragModel
from .polar import ParabolicDragPolar

__all__ = [
    "DragBuildReport",
    "DragCondition",
    "DragExpression",
    "DragModel",
    "GeometryDragModel",
    "ParabolicDragPolar",
    "TransonicGeometryDragModel",
    "build_drag_model",
    "build_drag_report",
    "build_parabolic_polar",
]
