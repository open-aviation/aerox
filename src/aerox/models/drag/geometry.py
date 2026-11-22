"""Condition-aware drag models constructed from aircraft geometry."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .components import (
    ParasiteDragComponent,
    fuselage_form_factor,
    fuselage_wetted_area,
    lifting_surface_form_factor,
    lifting_surface_wetted_area,
)
from .core import (
    CdShorthand,
    DragCondition,
    FloatArray,
    require_clean,
    resolve_condition,
)
from .wave import KornLockWaveDrag

if TYPE_CHECKING:
    from aerox.aircraft import Aircraft


@dataclass(frozen=True)
class GeometryDragBreakdown:
    """Detailed subcritical drag components."""

    component_parasite: Mapping[str, FloatArray]
    parasite_increments: Mapping[str, FloatArray]
    resolved_parasite: FloatArray
    increment_parasite: FloatArray
    parasite: FloatArray
    span_induced: FloatArray
    lift_dependent_profile: FloatArray
    total: FloatArray


@dataclass(frozen=True)
class GeometryDragModel(CdShorthand):
    """Condition-aware subcritical drag model derived from aircraft geometry.

    ``components`` vary with Reynolds and Mach number. ``parasite_increments``
    are named constant zero-lift coefficients (for example upsweep, gaps, or
    a published allowance) that are fixed at the reference condition.
    """

    reference_area_m2: float
    wing_span_m: float
    components: tuple[ParasiteDragComponent, ...]
    span_efficiency_factor: float
    profile_lift_drag_multiplier: float
    parasite_increments: tuple[tuple[str, float], ...]
    reference_reynolds_per_m: float
    parasite_scale: float = 1.0
    induced_scale: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.reference_area_m2,
                self.wing_span_m,
                self.span_efficiency_factor,
                self.profile_lift_drag_multiplier,
                self.reference_reynolds_per_m,
                self.parasite_scale,
                self.induced_scale,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("geometry drag parameters must be finite")
        if min(values[[0, 1, 2, 4, 5, 6]]) <= 0.0:
            raise ValueError(
                "geometry, Reynolds, and scale parameters must be positive"
            )
        if self.profile_lift_drag_multiplier < 0.0:
            raise ValueError("profile_lift_drag_multiplier must be non-negative")
        for _, increment in self.parasite_increments:
            if not np.isfinite(increment) or increment < 0.0:
                raise ValueError("parasite increments must be finite and non-negative")
        if not 0.3 <= self.span_efficiency_factor <= 1.2:
            raise ValueError("span_efficiency_factor is outside model support")
        if not self.components:
            raise ValueError("at least one parasite-drag component is required")
        names = [component.name for component in self.components]
        names.extend(name for name, _ in self.parasite_increments)
        if len(set(names)) != len(names):
            raise ValueError("component and increment names must be unique")

    @property
    def parasite_increment_coefficient(self) -> float:
        return float(sum(value for _, value in self.parasite_increments))

    @property
    def aspect_ratio(self) -> float:
        return self.wing_span_m**2 / self.reference_area_m2

    @property
    def span_induced_drag_factor(self) -> float:
        return self.induced_scale / (
            np.pi * self.aspect_ratio * self.span_efficiency_factor
        )

    def breakdown(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> GeometryDragBreakdown:
        condition = resolve_condition(
            mach, reynolds_per_m, condition, h=h, dT=dT
        ).with_default_reynolds(self.reference_reynolds_per_m)
        lift, mach, unit_reynolds, flap, gear, spoiler = condition.broadcast_with(cl)
        require_clean(flap, gear, spoiler)
        component_parasite = {
            component.name: self.parasite_scale
            * component.drag_coefficient(
                unit_reynolds,
                mach,
                self.reference_area_m2,
            )
            for component in self.components
        }
        resolved = np.sum(np.stack(tuple(component_parasite.values())), axis=0)
        parasite_increments = {
            name: np.full_like(resolved, self.parasite_scale * value)
            for name, value in self.parasite_increments
        }
        increment = np.full_like(
            resolved, self.parasite_scale * self.parasite_increment_coefficient
        )
        parasite = resolved + increment
        span_induced = self.span_induced_drag_factor * lift**2
        profile = (
            self.induced_scale * self.profile_lift_drag_multiplier * parasite * lift**2
        )
        total = parasite + span_induced + profile
        return GeometryDragBreakdown(
            component_parasite=component_parasite,
            parasite_increments=parasite_increments,
            resolved_parasite=resolved,
            increment_parasite=increment,
            parasite=parasite,
            span_induced=span_induced,
            lift_dependent_profile=profile,
            total=total,
        )

    def drag_coefficient(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
        ).total

    def _drag_coefficient_scalar(self, cl: float, condition: DragCondition) -> float:
        condition = condition.with_default_reynolds(self.reference_reynolds_per_m)
        values = (
            cl,
            condition.mach,
            condition.reynolds_per_m,
            condition.flap_angle_rad,
            condition.gear_extension_fraction,
            condition.spoiler_deflection_rad,
        )
        if any(np.ndim(value) != 0 for value in values):
            raise ValueError("scalar drag evaluation requires scalar condition values")
        lift, mach, unit_reynolds, flap, gear, spoiler = (
            float(value) for value in values
        )
        if not np.isfinite(lift):
            raise ValueError("lift coefficient (cl) must be finite")
        if not np.isfinite(mach) or mach < 0.0:
            raise ValueError("mach must be finite and non-negative")
        if not np.isfinite(unit_reynolds) or unit_reynolds <= 0.0:
            raise ValueError("reynolds_per_m must be finite and positive")
        if any(not np.isfinite(value) for value in (flap, gear, spoiler)):
            raise ValueError("configuration values must be finite")
        if flap < 0.0 or not 0.0 <= gear <= 1.0 or spoiler < 0.0:
            raise ValueError("configuration values are outside physical bounds")
        if any(value > 0.0 for value in (flap, gear, spoiler)):
            require_clean(np.asarray(flap), np.asarray(gear), np.asarray(spoiler))
        parasite = self.parasite_scale * (
            self.parasite_increment_coefficient
            + sum(
                component.drag_coefficient_scalar(
                    unit_reynolds,
                    mach,
                    self.reference_area_m2,
                )
                for component in self.components
            )
        )
        span_induced = self.span_induced_drag_factor * lift**2
        profile = (
            self.induced_scale * self.profile_lift_drag_multiplier * parasite * lift**2
        )
        return float(parasite + span_induced + profile)


@dataclass(frozen=True)
class TransonicGeometryDragBreakdown:
    """Geometry-derived drag components plus the canonical wave increment."""

    component_parasite: Mapping[str, FloatArray]
    parasite_increments: Mapping[str, FloatArray]
    resolved_parasite: FloatArray
    increment_parasite: FloatArray
    parasite: FloatArray
    span_induced: FloatArray
    lift_dependent_profile: FloatArray
    wave: FloatArray
    critical_mach: FloatArray
    total: FloatArray


@dataclass(frozen=True)
class TransonicGeometryDragModel(CdShorthand):
    """Condition-aware geometry model composed with Korn/Lock wave drag."""

    base_model: GeometryDragModel
    wave_drag: KornLockWaveDrag
    minimum_lift_coefficient: float = -0.5
    maximum_lift_coefficient: float = 2.5
    minimum_mach: float = 0.0
    maximum_mach: float = 0.95

    def __post_init__(self) -> None:
        values = np.asarray(
            [
                self.minimum_lift_coefficient,
                self.maximum_lift_coefficient,
                self.minimum_mach,
                self.maximum_mach,
            ],
            dtype=float,
        )
        if np.any(~np.isfinite(values)):
            raise ValueError("transonic drag bounds must be finite")
        if self.minimum_lift_coefficient >= self.maximum_lift_coefficient:
            raise ValueError("lift coefficient bounds are invalid")
        if self.minimum_mach < 0.0 or self.minimum_mach >= self.maximum_mach:
            raise ValueError("Mach bounds are invalid")

    def _validate_envelope(self, lift: FloatArray, mach: FloatArray) -> None:
        if np.any(
            (lift < self.minimum_lift_coefficient)
            | (lift > self.maximum_lift_coefficient)
        ):
            raise ValueError("lift coefficient is outside transonic-model support")
        if np.any((mach < self.minimum_mach) | (mach > self.maximum_mach)):
            raise ValueError("Mach is outside transonic-model support")

    def critical_mach(self, lift_coefficient: ArrayLike) -> FloatArray:
        return self.wave_drag.critical_mach(lift_coefficient)

    def breakdown(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> TransonicGeometryDragBreakdown:
        condition = resolve_condition(mach, reynolds_per_m, condition, h=h, dT=dT)
        lift, mach, *_ = condition.broadcast_with(cl)
        self._validate_envelope(lift, mach)
        base = self.base_model.breakdown(lift, condition=condition)
        wave, critical = self.wave_drag.evaluate(mach, lift)
        return TransonicGeometryDragBreakdown(
            component_parasite=base.component_parasite,
            parasite_increments=base.parasite_increments,
            resolved_parasite=base.resolved_parasite,
            increment_parasite=base.increment_parasite,
            parasite=base.parasite,
            span_induced=base.span_induced,
            lift_dependent_profile=base.lift_dependent_profile,
            wave=wave,
            critical_mach=critical,
            total=base.total + wave,
        )

    def drag_coefficient(
        self,
        cl: ArrayLike,
        mach: ArrayLike | None = None,
        reynolds_per_m: ArrayLike | None = None,
        *,
        h: ArrayLike | None = None,
        dT: ArrayLike | None = None,
        condition: DragCondition | None = None,
    ) -> FloatArray:
        return self.breakdown(
            cl, mach, reynolds_per_m, h=h, dT=dT, condition=condition
        ).total

    def _drag_coefficient_scalar(self, cl: float, condition: DragCondition) -> float:
        if np.ndim(condition.mach) != 0:
            raise ValueError("scalar drag evaluation requires scalar Mach")
        lift = float(cl)
        mach = float(condition.mach)
        self._validate_envelope(np.asarray(lift), np.asarray(mach))
        base = self.base_model._drag_coefficient_scalar(lift, condition)
        wave = self.wave_drag.drag_coefficient_increment_scalar(mach, lift)
        return float(base + wave)


def _required_positive(value: float | None, path: str) -> float:
    if value is None or not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"aircraft.{path} must be available and positive")
    return float(value)


def _tail_component_from_dimensions(
    name: str,
    area_m2: float | None,
    mean_aerodynamic_chord_m: float | None,
    *,
    thickness_to_chord_ratio: float,
    interference_factor: float = 1.0,
) -> ParasiteDragComponent | None:
    if area_m2 is None and mean_aerodynamic_chord_m is None:
        return None
    if area_m2 is None or mean_aerodynamic_chord_m is None:
        raise ValueError(f"{name} area and chord must be supplied together")
    area = _required_positive(area_m2, f"{name}.area")
    chord = _required_positive(mean_aerodynamic_chord_m, f"{name}.mac")
    return ParasiteDragComponent(
        name=name,
        wetted_area_m2=lifting_surface_wetted_area(area, thickness_to_chord_ratio),
        characteristic_length_m=chord,
        form_factor=lifting_surface_form_factor(thickness_to_chord_ratio),
        interference_factor=interference_factor,
    )


def geometry_drag_model_from_dimensions(
    *,
    wing_area_m2: float,
    wing_span_m: float,
    wing_mean_aerodynamic_chord_m: float,
    wing_thickness_to_chord_ratio: float,
    wing_sweep_rad: float,
    fuselage_length_m: float,
    fuselage_width_m: float,
    fuselage_height_m: float,
    reference_mach: float,
    reference_reynolds_per_m: float,
    span_efficiency_factor: float,
    horizontal_tail_area_m2: float | None = None,
    horizontal_tail_mean_aerodynamic_chord_m: float | None = None,
    vertical_tail_area_m2: float | None = None,
    vertical_tail_mean_aerodynamic_chord_m: float | None = None,
    tail_thickness_to_chord_ratio: float = 0.10,
    wing_wetted_planform_area_m2: float | None = None,
    tail_interference_factor: float = 1.0,
    additional_components: tuple[ParasiteDragComponent, ...] = (),
    additional_drag_areas_m2: tuple[tuple[str, float], ...] = (),
    allowance_fraction: float = 0.10,
    allowance_name: str = "unresolved",
    parasite_scale: float = 1.0,
    induced_scale: float = 1.0,
) -> GeometryDragModel:
    """Build the physical model once all geometry dimensions are resolved.

    This is the common construction boundary for directly sourced geometry and
    aircraft records whose missing dimensions have first been scaled explicitly.
    The wing wetted area uses ``wing_wetted_planform_area_m2`` (for example
    the exposed area) and falls back to the gross reference area.
    ``additional_drag_areas_m2`` are constant drag areas ``D/q`` divided by
    the reference area. The named allowance is ``allowance_fraction`` times
    the sum of all computed terms at the reference condition.
    """

    if not 0.0 <= allowance_fraction <= 1.0:
        raise ValueError("allowance_fraction must lie between zero and one")
    wing_area = _required_positive(wing_area_m2, "wing.area")
    wing_span = _required_positive(wing_span_m, "wing.span")
    wing_chord = _required_positive(wing_mean_aerodynamic_chord_m, "wing.mac")
    wing_thickness = _required_positive(wing_thickness_to_chord_ratio, "wing.t_c")
    fuselage_length = _required_positive(fuselage_length_m, "fuselage.length")
    fuselage_width = _required_positive(fuselage_width_m, "fuselage.width")
    fuselage_height = _required_positive(fuselage_height_m, "fuselage.height")
    wing_planform = (
        wing_area
        if wing_wetted_planform_area_m2 is None
        else _required_positive(wing_wetted_planform_area_m2, "wing.exposed_area")
    )
    components: list[ParasiteDragComponent] = [
        ParasiteDragComponent(
            name="wing",
            wetted_area_m2=lifting_surface_wetted_area(wing_planform, wing_thickness),
            characteristic_length_m=wing_chord,
            form_factor=lifting_surface_form_factor(wing_thickness),
        ),
        ParasiteDragComponent(
            name="fuselage",
            wetted_area_m2=fuselage_wetted_area(
                fuselage_length, fuselage_width, fuselage_height
            ),
            characteristic_length_m=fuselage_length,
            form_factor=fuselage_form_factor(
                fuselage_length, fuselage_width, fuselage_height
            ),
        ),
    ]
    for name, area, chord in (
        (
            "horizontal_tail",
            horizontal_tail_area_m2,
            horizontal_tail_mean_aerodynamic_chord_m,
        ),
        (
            "vertical_tail",
            vertical_tail_area_m2,
            vertical_tail_mean_aerodynamic_chord_m,
        ),
    ):
        component = _tail_component_from_dimensions(
            name,
            area,
            chord,
            thickness_to_chord_ratio=tail_thickness_to_chord_ratio,
            interference_factor=tail_interference_factor,
        )
        if component is not None:
            components.append(component)
    components.extend(additional_components)
    increments = [
        (name, float(area) / wing_area) for name, area in additional_drag_areas_m2
    ]

    computed_reference = sum(
        component.drag_coefficient_scalar(
            reference_reynolds_per_m,
            reference_mach,
            wing_area,
        )
        for component in components
    ) + sum(value for _, value in increments)
    increments.append((allowance_name, allowance_fraction * computed_reference))
    profile_multiplier = 0.80 * (1.0 - 0.53 * np.cos(wing_sweep_rad))
    return GeometryDragModel(
        reference_area_m2=wing_area,
        wing_span_m=wing_span,
        components=tuple(components),
        span_efficiency_factor=span_efficiency_factor,
        profile_lift_drag_multiplier=float(profile_multiplier),
        parasite_increments=tuple(increments),
        reference_reynolds_per_m=(reference_reynolds_per_m),
        parasite_scale=parasite_scale,
        induced_scale=induced_scale,
    )


def geometry_drag_model_from_aircraft(
    aircraft: Aircraft,
    *,
    reference_reynolds_per_m: float = 5.05e6,
    span_efficiency_factor: float = 0.95,
    tail_thickness_to_chord_ratio: float = 0.10,
    unresolved_fraction_of_resolved_drag: float = 0.10,
    parasite_scale: float = 1.0,
    induced_scale: float = 1.0,
) -> GeometryDragModel:
    """Build a transparent clean-aircraft model from available Aerox geometry.

    This direct path uses the gross wing area and one lumped unresolved term
    for all components not represented here. The term is computed at the
    reference Reynolds condition and remains explicit in all breakdowns.
    """

    if aircraft.wing is None or aircraft.fuselage is None:
        raise ValueError("aircraft wing and fuselage geometry are required")
    wing_area = _required_positive(aircraft.wing.area, "wing.area")
    wing_span = _required_positive(aircraft.wing.span, "wing.span")
    wing_chord = _required_positive(aircraft.wing.mac, "wing.mac")
    wing_thickness = _required_positive(aircraft.wing.t_c, "wing.t_c")
    fuselage_length = _required_positive(aircraft.fuselage.length, "fuselage.length")
    fuselage_width = _required_positive(aircraft.fuselage.width, "fuselage.width")
    fuselage_height = _required_positive(aircraft.fuselage.height, "fuselage.height")
    sweep_rad = np.deg2rad(float(aircraft.wing.sweep or 0.0))
    reference_mach = (
        float(aircraft.kinematic.cruise_mach or 0.78)
        if aircraft.kinematic is not None
        else 0.78
    )
    horizontal_tail = aircraft.horizontal_tail
    vertical_tail = aircraft.vertical_tail
    return geometry_drag_model_from_dimensions(
        wing_area_m2=wing_area,
        wing_span_m=wing_span,
        wing_mean_aerodynamic_chord_m=wing_chord,
        wing_thickness_to_chord_ratio=wing_thickness,
        wing_sweep_rad=float(sweep_rad),
        fuselage_length_m=fuselage_length,
        fuselage_width_m=fuselage_width,
        fuselage_height_m=fuselage_height,
        horizontal_tail_area_m2=(
            None if horizontal_tail is None else horizontal_tail.area
        ),
        horizontal_tail_mean_aerodynamic_chord_m=(
            None if horizontal_tail is None else horizontal_tail.mac
        ),
        vertical_tail_area_m2=(None if vertical_tail is None else vertical_tail.area),
        vertical_tail_mean_aerodynamic_chord_m=(
            None if vertical_tail is None else vertical_tail.mac
        ),
        reference_mach=reference_mach,
        reference_reynolds_per_m=(reference_reynolds_per_m),
        span_efficiency_factor=span_efficiency_factor,
        tail_thickness_to_chord_ratio=tail_thickness_to_chord_ratio,
        allowance_fraction=unresolved_fraction_of_resolved_drag,
        parasite_scale=parasite_scale,
        induced_scale=induced_scale,
    )
