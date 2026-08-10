"""Leakage-safe, phase-conditioned CAS/Mach intent primitives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from aerox import aero

FloatArray = NDArray[np.float64]
StringArray = NDArray[np.str_]
KNOT_TO_MPS = 1852.0 / 3600.0
FT_TO_M = 0.3048

PHASES = ("CLIMB", "LEVEL", "DESCENT")
PHASE_CONSTANT_FEATURES = ("bias", "phase_level", "phase_descent")
AFFINE_SELECTOR_FEATURES = (
    "bias",
    "selector",
    "altitude_20kft",
    "mass_offset_10t",
    "phase_level",
    "phase_descent",
)


def vertical_phase(vertical_rate_ft_min: FloatArray) -> StringArray:
    """Map an aligned planned vertical rate to broad FMS phases."""

    rate = np.asarray(vertical_rate_ft_min, dtype=float)
    if rate.ndim != 1 or np.any(~np.isfinite(rate)):
        raise ValueError("vertical rate must be a finite one-dimensional array")
    return np.where(rate > 250.0, "CLIMB", np.where(rate < -250.0, "DESCENT", "LEVEL"))


def speed_intent_design(
    *,
    selector: FloatArray,
    pressure_altitude_ft: FloatArray,
    mass_kg: FloatArray,
    phase: StringArray,
    speed_kind: str,
    feature_set: str,
) -> FloatArray:
    """Build a scaled design matrix from generation-time quantities only."""

    selected = np.asarray(selector, dtype=float)
    altitude = np.asarray(pressure_altitude_ft, dtype=float)
    mass = np.asarray(mass_kg, dtype=float)
    phase_array = np.asarray(phase, dtype=str)
    if not (selected.shape == altitude.shape == mass.shape == phase_array.shape):
        raise ValueError("speed-intent inputs must have equal shapes")
    if selected.ndim != 1 or np.any(~np.isfinite(selected + altitude + mass)):
        raise ValueError("speed-intent numeric inputs must be finite vectors")
    if not np.all(np.isin(phase_array, PHASES)):
        raise ValueError(f"phase must contain only {PHASES}")
    if speed_kind not in {"CAS", "MACH"}:
        raise ValueError("speed_kind must be CAS or MACH")
    if feature_set not in {"phase_constant", "affine_selector"}:
        raise ValueError("unknown speed-intent feature set")

    columns = [
        np.ones(len(selected)),
        (phase_array == "LEVEL").astype(float),
        (phase_array == "DESCENT").astype(float),
    ]
    if feature_set == "phase_constant":
        return np.column_stack(columns)

    selector_scale = 50.0 if speed_kind == "CAS" else 0.1
    selector_center = 275.0 if speed_kind == "CAS" else 0.75
    return np.column_stack(
        [
            np.ones(len(selected)),
            (selected - selector_center) / selector_scale,
            altitude / 20_000.0,
            (mass - 60_000.0) / 10_000.0,
            (phase_array == "LEVEL").astype(float),
            (phase_array == "DESCENT").astype(float),
        ]
    )


@dataclass(frozen=True)
class SpeedIntentPrediction:
    """CAS and Mach targets plus the response-blind active schedule branch."""

    cas_target_kts: FloatArray
    mach_target: FloatArray
    unconstrained_crossover_altitude_ft: FloatArray
    crossover_altitude_ft: FloatArray
    active_kind: StringArray
    active_cas_equivalent_kts: FloatArray

    def __post_init__(self) -> None:
        numeric_names = (
            "cas_target_kts",
            "mach_target",
            "unconstrained_crossover_altitude_ft",
            "crossover_altitude_ft",
            "active_cas_equivalent_kts",
        )
        numeric = {
            name: np.asarray(getattr(self, name), dtype=float) for name in numeric_names
        }
        active_kind = np.asarray(self.active_kind, dtype=str)
        shape = numeric["cas_target_kts"].shape
        if (
            len(shape) != 1
            or any(array.shape != shape for array in numeric.values())
            or active_kind.shape != shape
            or any(np.any(~np.isfinite(array)) for array in numeric.values())
            or not np.all(np.isin(active_kind, ("CAS", "MACH")))
        ):
            raise ValueError(
                "speed-intent prediction arrays must be finite and aligned"
            )
        for name, array in numeric.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "active_kind", active_kind)


@dataclass(frozen=True)
class ModeConditionedSpeedIntent:
    """Frozen phase-conditioned target model selected outside held-out flights."""

    structure: str
    cas_coefficients: FloatArray
    mach_coefficients: FloatArray
    minimum_cas_kts: float = 180.0
    maximum_cas_kts: float = 350.0
    minimum_mach: float = 0.50
    maximum_mach: float = 0.86
    minimum_crossover_altitude_ft: float = 25_000.0
    maximum_crossover_altitude_ft: float = 35_000.0

    def __post_init__(self) -> None:
        if self.structure not in {"phase_constant", "selector", "affine_selector"}:
            raise ValueError("unknown speed-intent structure")
        cas = np.asarray(self.cas_coefficients, dtype=float)
        mach = np.asarray(self.mach_coefficients, dtype=float)
        expected = 6 if self.structure == "affine_selector" else 3
        if self.structure == "selector":
            expected = 0
        if cas.shape != (expected,) or mach.shape != (expected,):
            raise ValueError("coefficient vector does not match model structure")
        if np.any(~np.isfinite(cas)) or np.any(~np.isfinite(mach)):
            raise ValueError("speed-intent coefficients must be finite")
        object.__setattr__(self, "cas_coefficients", cas)
        object.__setattr__(self, "mach_coefficients", mach)
        bounds = (
            self.minimum_cas_kts,
            self.maximum_cas_kts,
            self.minimum_mach,
            self.maximum_mach,
            self.minimum_crossover_altitude_ft,
            self.maximum_crossover_altitude_ft,
        )
        if any(not np.isfinite(value) for value in bounds):
            raise ValueError("speed-intent bounds must be finite")
        if not self.minimum_cas_kts < self.maximum_cas_kts:
            raise ValueError("CAS bounds are reversed")
        if not 0.0 < self.minimum_mach < self.maximum_mach < 1.0:
            raise ValueError("Mach bounds must lie between zero and one")
        if not (
            0.0
            < self.minimum_crossover_altitude_ft
            < self.maximum_crossover_altitude_ft
            < 60_000.0
        ):
            raise ValueError("crossover bounds must be ordered operational altitudes")

    @property
    def feature_set(self) -> str:
        if self.structure == "selector":
            return "selector"
        return self.structure

    def predict(
        self,
        *,
        selected_cas_kts: FloatArray,
        selected_mach: FloatArray,
        pressure_altitude_ft: FloatArray,
        mass_kg: FloatArray,
        phase: StringArray,
    ) -> SpeedIntentPrediction:
        """Predict a conventional CAS/Mach schedule without aircraft response."""

        selected_cas = np.asarray(selected_cas_kts, dtype=float)
        selected_mach_array = np.asarray(selected_mach, dtype=float)
        altitude_ft = np.asarray(pressure_altitude_ft, dtype=float)
        mass = np.asarray(mass_kg, dtype=float)
        phase_array = np.asarray(phase, dtype=str)

        if not (
            selected_cas.shape
            == selected_mach_array.shape
            == altitude_ft.shape
            == mass.shape
            == phase_array.shape
        ):
            raise ValueError("speed-intent prediction inputs must have equal shapes")
        if selected_cas.ndim != 1 or np.any(
            ~np.isfinite(selected_cas + selected_mach_array + altitude_ft + mass)
        ):
            raise ValueError("speed-intent prediction inputs must be finite vectors")

        if self.structure == "selector":
            cas_target = selected_cas.copy()
            mach_target = selected_mach_array.copy()
            # Validate common inputs through the same design helper.
            speed_intent_design(
                selector=selected_cas,
                pressure_altitude_ft=altitude_ft,
                mass_kg=mass,
                phase=phase_array,
                speed_kind="CAS",
                feature_set="phase_constant",
            )
        else:
            cas_design = speed_intent_design(
                selector=selected_cas,
                pressure_altitude_ft=altitude_ft,
                mass_kg=mass,
                phase=phase_array,
                speed_kind="CAS",
                feature_set=self.structure,
            )
            mach_design = speed_intent_design(
                selector=selected_mach_array,
                pressure_altitude_ft=altitude_ft,
                mass_kg=mass,
                phase=phase_array,
                speed_kind="MACH",
                feature_set=self.structure,
            )
            cas_target = cas_design @ np.asarray(self.cas_coefficients, dtype=float)
            mach_target = mach_design @ np.asarray(self.mach_coefficients, dtype=float)

        cas_target = np.clip(cas_target, self.minimum_cas_kts, self.maximum_cas_kts)
        mach_target = np.clip(mach_target, self.minimum_mach, self.maximum_mach)
        unconstrained_crossover_ft = (
            np.asarray(
                aero.crossover_alt(cas_target * KNOT_TO_MPS, mach_target),
                dtype=float,
            )
            / FT_TO_M
        )
        crossover_ft = np.clip(
            unconstrained_crossover_ft,
            self.minimum_crossover_altitude_ft,
            self.maximum_crossover_altitude_ft,
        )
        use_mach = altitude_ft >= crossover_ft
        mach_cas = (
            np.asarray(aero.mach2cas(mach_target, altitude_ft * FT_TO_M), dtype=float)
            / KNOT_TO_MPS
        )
        active_cas = np.where(use_mach, mach_cas, cas_target)
        return SpeedIntentPrediction(
            cas_target_kts=cas_target,
            mach_target=mach_target,
            unconstrained_crossover_altitude_ft=unconstrained_crossover_ft,
            crossover_altitude_ft=crossover_ft,
            active_kind=np.where(use_mach, "MACH", "CAS"),
            active_cas_equivalent_kts=active_cas,
        )
