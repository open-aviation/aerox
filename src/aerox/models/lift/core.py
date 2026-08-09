"""Low-order aerodynamic maps for progressive aircraft-model commissioning."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]

LIFT_FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "linear_alpha": ("constant", "alpha_centered"),
    "linear_alpha_mach": (
        "constant",
        "alpha_centered",
        "mach_centered",
        "alpha_mach_interaction",
    ),
    "quadratic_alpha_mach": (
        "constant",
        "alpha_centered",
        "alpha_centered_squared",
        "mach_centered",
        "alpha_mach_interaction",
    ),
}


def effective_lift_design(
    angle_of_attack_rad: ArrayLike,
    mach: ArrayLike,
    *,
    feature_set: str,
) -> FloatArray:
    """Return a centered design matrix for an effective clean lift map."""
    if feature_set not in LIFT_FEATURE_SETS:
        raise ValueError(f"unknown lift feature set: {feature_set}")
    alpha = np.asarray(angle_of_attack_rad, dtype=float)
    mach_array = np.asarray(mach, dtype=float)
    if alpha.ndim != 1 or mach_array.ndim != 1 or alpha.shape != mach_array.shape:
        raise ValueError("angle_of_attack_rad and mach must be equal-length vectors")
    if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(mach_array)):
        raise ValueError("angle_of_attack_rad and mach must be finite")
    # Degree scaling keeps the low-order fit numerically well conditioned and
    # makes the slope coefficient directly interpretable as CL per degree.
    alpha_centered = np.rad2deg(alpha) - 4.0
    mach_centered = mach_array - 0.6
    available = {
        "constant": np.ones_like(alpha),
        "alpha_centered": alpha_centered,
        "alpha_centered_squared": alpha_centered**2,
        "mach_centered": mach_centered,
        "alpha_mach_interaction": alpha_centered * mach_centered,
    }
    return np.column_stack([available[name] for name in LIFT_FEATURE_SETS[feature_set]])


@dataclass(frozen=True)
class EffectiveLiftMap:
    """Effective clean-configuration lift coefficient map."""

    feature_set: str
    coefficients: FloatArray
    minimum_lift_coefficient: float = -0.5
    maximum_lift_coefficient: float = 2.0

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=float)
        expected = len(LIFT_FEATURE_SETS.get(self.feature_set, ()))
        if coefficients.ndim != 1 or len(coefficients) != expected:
            raise ValueError("coefficient count does not match lift feature set")
        if np.any(~np.isfinite(coefficients)):
            raise ValueError("coefficients must be finite")
        if self.minimum_lift_coefficient >= self.maximum_lift_coefficient:
            raise ValueError("lift coefficient bounds are invalid")
        object.__setattr__(self, "coefficients", coefficients)

    def raw_lift_coefficient(
        self, angle_of_attack_rad: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        design = effective_lift_design(
            angle_of_attack_rad,
            mach,
            feature_set=self.feature_set,
        )
        return design @ self.coefficients

    def raw_lift_coefficient_scalar(
        self, angle_of_attack_rad: float, mach: float
    ) -> float:
        """Scalar fast path for numerical integration."""
        alpha_centered = np.rad2deg(angle_of_attack_rad) - 4.0
        mach_centered = mach - 0.6
        available = {
            "constant": 1.0,
            "alpha_centered": alpha_centered,
            "alpha_centered_squared": alpha_centered**2,
            "mach_centered": mach_centered,
            "alpha_mach_interaction": alpha_centered * mach_centered,
        }
        return float(
            sum(
                coefficient * available[name]
                for coefficient, name in zip(
                    self.coefficients,
                    LIFT_FEATURE_SETS[self.feature_set],
                    strict=True,
                )
            )
        )

    def lift_coefficient(
        self, angle_of_attack_rad: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        return np.clip(
            self.raw_lift_coefficient(angle_of_attack_rad, mach),
            self.minimum_lift_coefficient,
            self.maximum_lift_coefficient,
        )

    def lift_coefficient_scalar(self, angle_of_attack_rad: float, mach: float) -> float:
        return float(
            np.clip(
                self.raw_lift_coefficient_scalar(angle_of_attack_rad, mach),
                self.minimum_lift_coefficient,
                self.maximum_lift_coefficient,
            )
        )


@dataclass(frozen=True)
class StaticAerodynamicCoefficients:
    """Five inviscid coefficients using VSPAERO's stability-output names.

    Drag is deliberately absent. A thin-surface stability prior must be paired
    with a separately sourced viscous drag model. These values have not yet
    been transformed into Aerox's eventual 6-DOF body-axis force convention.
    """

    lift: float
    side_force: float
    roll_moment: float
    pitch_moment: float
    yaw_moment: float


@dataclass(frozen=True)
class TabulatedStaticStabilityPrior:
    """Mach-interpolated local-linear static aerodynamic derivative prior.

    Alpha and beta derivatives are per radian in the VSPAERO `.stab`
    convention. The model rejects extrapolation beyond its explicitly supplied
    Mach/angle envelope. Body-rate and control-surface derivatives are outside
    this static interface.
    """

    mach: FloatArray
    reference_alpha_rad: float
    reference_lift: FloatArray
    reference_side_force: FloatArray
    reference_roll_moment: FloatArray
    reference_pitch_moment: FloatArray
    reference_yaw_moment: FloatArray
    lift_alpha: FloatArray
    side_force_beta: FloatArray
    roll_moment_beta: FloatArray
    pitch_moment_alpha: FloatArray
    yaw_moment_beta: FloatArray
    minimum_alpha_rad: float
    maximum_alpha_rad: float
    minimum_beta_rad: float
    maximum_beta_rad: float

    def __post_init__(self) -> None:
        mach = np.asarray(self.mach, dtype=float)
        names = (
            "reference_lift",
            "reference_side_force",
            "reference_roll_moment",
            "reference_pitch_moment",
            "reference_yaw_moment",
            "lift_alpha",
            "side_force_beta",
            "roll_moment_beta",
            "pitch_moment_alpha",
            "yaw_moment_beta",
        )
        if mach.ndim != 1 or len(mach) < 2 or np.any(~np.isfinite(mach)):
            raise ValueError("mach must contain at least two finite points")
        if np.any(np.diff(mach) <= 0.0):
            raise ValueError("mach points must be strictly increasing")
        object.__setattr__(self, "mach", mach)
        for name in names:
            values = np.asarray(getattr(self, name), dtype=float)
            if values.shape != mach.shape or np.any(~np.isfinite(values)):
                raise ValueError(f"{name} must be finite and match mach")
            object.__setattr__(self, name, values)
        scalar_values = (
            self.reference_alpha_rad,
            self.minimum_alpha_rad,
            self.maximum_alpha_rad,
            self.minimum_beta_rad,
            self.maximum_beta_rad,
        )
        if any(not np.isfinite(value) for value in scalar_values):
            raise ValueError("angle references and bounds must be finite")
        if self.minimum_alpha_rad >= self.maximum_alpha_rad:
            raise ValueError("alpha bounds are invalid")
        if self.minimum_beta_rad >= self.maximum_beta_rad:
            raise ValueError("beta bounds are invalid")
        if (
            not self.minimum_alpha_rad
            <= self.reference_alpha_rad
            <= self.maximum_alpha_rad
        ):
            raise ValueError("reference alpha is outside the alpha bounds")

    @classmethod
    def from_vspaero_csv(
        cls,
        path: str | Path,
        *,
        mesh: str = "medium",
        alpha_bounds_deg: tuple[float, float] = (-1.0, 6.0),
        beta_bounds_deg: tuple[float, float] = (-3.0, 3.0),
    ) -> TabulatedStaticStabilityPrior:
        """Load released static columns from an Aerox VSPAERO results table."""

        with Path(path).open(newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["mesh"] == mesh]
        if len(rows) < 2:
            raise ValueError(f"fewer than two {mesh!r} mesh rows in {path}")
        rows.sort(key=lambda row: float(row["mach"]))
        alpha_values = np.array([float(row["alpha_deg"]) for row in rows])
        if not np.allclose(alpha_values, alpha_values[0], atol=1e-12):
            raise ValueError("VSPAERO reference alpha must be common across Mach")

        def column(name: str) -> FloatArray:
            return np.array([float(row[name]) for row in rows], dtype=float)

        return cls(
            mach=column("mach"),
            reference_alpha_rad=float(np.deg2rad(alpha_values[0])),
            reference_lift=column("CL"),
            reference_side_force=column("CS"),
            reference_roll_moment=column("CMl"),
            reference_pitch_moment=column("CMm"),
            reference_yaw_moment=column("CMn"),
            lift_alpha=column("CL_alpha"),
            side_force_beta=column("CS_beta"),
            roll_moment_beta=column("CMl_beta"),
            pitch_moment_alpha=column("CMm_alpha"),
            yaw_moment_beta=column("CMn_beta"),
            minimum_alpha_rad=float(np.deg2rad(alpha_bounds_deg[0])),
            maximum_alpha_rad=float(np.deg2rad(alpha_bounds_deg[1])),
            minimum_beta_rad=float(np.deg2rad(beta_bounds_deg[0])),
            maximum_beta_rad=float(np.deg2rad(beta_bounds_deg[1])),
        )

    def _interpolate(self, values: FloatArray, mach: float) -> float:
        if not np.isfinite(mach):
            raise ValueError("mach must be finite")
        if mach < self.mach[0] or mach > self.mach[-1]:
            raise ValueError(
                f"mach {mach:.4g} is outside [{self.mach[0]:.4g}, {self.mach[-1]:.4g}]"
            )
        return float(np.interp(mach, self.mach, values))

    def coefficients(
        self, *, angle_of_attack_rad: float, sideslip_rad: float, mach: float
    ) -> StaticAerodynamicCoefficients:
        """Evaluate the local-linear VSPAERO prior without a drag coefficient.

        ``sideslip_rad`` must follow VSPAERO's beta convention. This method does
        not claim that convention as the future Aerox body-axis convention.
        """

        if not np.isfinite(angle_of_attack_rad) or not np.isfinite(sideslip_rad):
            raise ValueError("angle_of_attack_rad and sideslip_rad must be finite")
        if not self.minimum_alpha_rad <= angle_of_attack_rad <= self.maximum_alpha_rad:
            raise ValueError("angle_of_attack_rad is outside the static-prior envelope")
        if not self.minimum_beta_rad <= sideslip_rad <= self.maximum_beta_rad:
            raise ValueError("sideslip_rad is outside the static-prior envelope")
        delta_alpha = angle_of_attack_rad - self.reference_alpha_rad
        return StaticAerodynamicCoefficients(
            lift=self._interpolate(self.reference_lift, mach)
            + self._interpolate(self.lift_alpha, mach) * delta_alpha,
            side_force=self._interpolate(self.reference_side_force, mach)
            + self._interpolate(self.side_force_beta, mach) * sideslip_rad,
            roll_moment=self._interpolate(self.reference_roll_moment, mach)
            + self._interpolate(self.roll_moment_beta, mach) * sideslip_rad,
            pitch_moment=self._interpolate(self.reference_pitch_moment, mach)
            + self._interpolate(self.pitch_moment_alpha, mach) * delta_alpha,
            yaw_moment=self._interpolate(self.reference_yaw_moment, mach)
            + self._interpolate(self.yaw_moment_beta, mach) * sideslip_rad,
        )


@dataclass(frozen=True)
class TabulatedLiftSurface:
    """Bounded bilinear ``CL(alpha, Mach)`` surface from VSPAERO base solutions.

    This interface deliberately carries lift only. It does not imply that
    VSPAERO's inviscid drag is suitable for Aerox, and it does not use the
    numerically fragile local pitching-moment derivative at the edge of the
    sampled alpha/Mach envelope.
    """

    mach: FloatArray
    angle_of_attack_rad: FloatArray
    lift_grid: FloatArray

    def __post_init__(self) -> None:
        mach = np.asarray(self.mach, dtype=float)
        alpha = np.asarray(self.angle_of_attack_rad, dtype=float)
        lift = np.asarray(self.lift_grid, dtype=float)
        if mach.ndim != 1 or len(mach) < 2 or np.any(np.diff(mach) <= 0.0):
            raise ValueError("mach must contain at least two increasing points")
        if alpha.ndim != 1 or len(alpha) < 2 or np.any(np.diff(alpha) <= 0.0):
            raise ValueError("angle_of_attack_rad must contain increasing points")
        if lift.shape != (len(mach), len(alpha)) or np.any(~np.isfinite(lift)):
            raise ValueError("lift_grid must be finite with shape (mach, alpha)")
        object.__setattr__(self, "mach", mach)
        object.__setattr__(self, "angle_of_attack_rad", alpha)
        object.__setattr__(self, "lift_grid", lift)

    @classmethod
    def from_vspaero_alpha_support_csv(cls, path: str | Path) -> TabulatedLiftSurface:
        """Load a complete rectangular alpha/Mach base-coefficient sweep."""

        with Path(path).open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        if not rows:
            raise ValueError(f"no VSPAERO alpha-support rows in {path}")
        mach = np.array(sorted({float(row["mach"]) for row in rows}), dtype=float)
        alpha_deg = np.array(
            sorted({float(row["alpha_deg"]) for row in rows}), dtype=float
        )
        indexed = {
            (float(row["mach"]), float(row["alpha_deg"])): float(row["CL"])
            for row in rows
        }
        expected = {(m, a) for m in mach for a in alpha_deg}
        if set(indexed) != expected:
            raise ValueError("VSPAERO alpha-support table is not a complete grid")
        lift = np.array(
            [[indexed[(m, a)] for a in alpha_deg] for m in mach], dtype=float
        )
        return cls(mach, np.deg2rad(alpha_deg), lift)

    @property
    def minimum_mach(self) -> float:
        return float(self.mach[0])

    @property
    def maximum_mach(self) -> float:
        return float(self.mach[-1])

    @property
    def minimum_alpha_rad(self) -> float:
        return float(self.angle_of_attack_rad[0])

    @property
    def maximum_alpha_rad(self) -> float:
        return float(self.angle_of_attack_rad[-1])

    def lift_coefficient(
        self, angle_of_attack_rad: ArrayLike, mach: ArrayLike
    ) -> FloatArray:
        """Evaluate the surface without extrapolating outside its support."""

        alpha = np.asarray(angle_of_attack_rad, dtype=float)
        mach_array = np.asarray(mach, dtype=float)
        if alpha.shape != mach_array.shape:
            raise ValueError("angle_of_attack_rad and mach must have equal shapes")
        if np.any(~np.isfinite(alpha)) or np.any(~np.isfinite(mach_array)):
            raise ValueError("angle_of_attack_rad and mach must be finite")
        if np.any((alpha < self.minimum_alpha_rad) | (alpha > self.maximum_alpha_rad)):
            raise ValueError("angle_of_attack_rad is outside the lift-surface envelope")
        if np.any((mach_array < self.minimum_mach) | (mach_array > self.maximum_mach)):
            raise ValueError("mach is outside the lift-surface envelope")

        flat_alpha = alpha.reshape(-1)
        flat_mach = mach_array.reshape(-1)
        mach_index = np.clip(
            np.searchsorted(self.mach, flat_mach, side="right") - 1,
            0,
            len(self.mach) - 2,
        )
        alpha_index = np.clip(
            np.searchsorted(self.angle_of_attack_rad, flat_alpha, side="right") - 1,
            0,
            len(self.angle_of_attack_rad) - 2,
        )
        mach_low = self.mach[mach_index]
        mach_high = self.mach[mach_index + 1]
        alpha_low = self.angle_of_attack_rad[alpha_index]
        alpha_high = self.angle_of_attack_rad[alpha_index + 1]
        mach_weight = (flat_mach - mach_low) / (mach_high - mach_low)
        alpha_weight = (flat_alpha - alpha_low) / (alpha_high - alpha_low)
        lower = (1.0 - alpha_weight) * self.lift_grid[
            mach_index, alpha_index
        ] + alpha_weight * self.lift_grid[mach_index, alpha_index + 1]
        upper = (1.0 - alpha_weight) * self.lift_grid[
            mach_index + 1, alpha_index
        ] + alpha_weight * self.lift_grid[mach_index + 1, alpha_index + 1]
        return ((1.0 - mach_weight) * lower + mach_weight * upper).reshape(alpha.shape)

    def lift_coefficient_scalar(self, angle_of_attack_rad: float, mach: float) -> float:
        return float(
            self.lift_coefficient(
                np.asarray([angle_of_attack_rad]), np.asarray([mach])
            )[0]
        )


@dataclass(frozen=True)
class CalibratedTabulatedLiftSurface:
    """Pitch-aware adapter for an identified bounded lift surface.

    The pitch scale applies to decoded pitch before subtracting the physical
    flight-path angle. ``registration_angle_offset_rad`` is an already-resolved
    tail-specific measurement nuisance for one plant instance.
    """

    surface: TabulatedLiftSurface
    pitch_scale: float
    decoded_pitch_bias_rad: float
    combined_offset_rad: float
    mach_offset_rad_per_mach: float
    registration_angle_offset_rad: float = 0.0
    mach_reference: float = 0.6

    def __post_init__(self) -> None:
        values = (
            self.pitch_scale,
            self.decoded_pitch_bias_rad,
            self.combined_offset_rad,
            self.mach_offset_rad_per_mach,
            self.registration_angle_offset_rad,
            self.mach_reference,
        )
        if any(not np.isfinite(value) for value in values):
            raise ValueError("calibrated lift-surface parameters must be finite")
        if self.pitch_scale <= 0.0:
            raise ValueError("pitch_scale must be positive")

    def effective_angle_of_attack_scalar(
        self, *, pitch_angle_rad: float, flight_path_angle_rad: float, mach: float
    ) -> float:
        return float(
            self.pitch_scale * (pitch_angle_rad - self.decoded_pitch_bias_rad)
            + self.combined_offset_rad
            + self.mach_offset_rad_per_mach * (mach - self.mach_reference)
            + self.registration_angle_offset_rad
            - flight_path_angle_rad
        )

    def angle_and_lift_coefficient_scalar(
        self, *, pitch_angle_rad: float, flight_path_angle_rad: float, mach: float
    ) -> tuple[float, float]:
        angle = self.effective_angle_of_attack_scalar(
            pitch_angle_rad=pitch_angle_rad,
            flight_path_angle_rad=flight_path_angle_rad,
            mach=mach,
        )
        return angle, self.surface.lift_coefficient_scalar(angle, mach)


@dataclass(frozen=True)
class GuardedHybridLiftModel:
    """Smoothly use a calibrated VSPAERO surface only inside confirmed support."""

    surface_model: CalibratedTabulatedLiftSurface
    empirical_fallback: EffectiveLiftMap
    lower_alpha_core_rad: float = float(np.deg2rad(-0.5))
    upper_alpha_core_rad: float = float(np.deg2rad(8.0))
    lower_mach_core: float = 0.23
    upper_mach_core: float = 0.75
    maximum_surface_weight: float = 1.0

    def __post_init__(self) -> None:
        surface = self.surface_model.surface
        if not surface.minimum_alpha_rad < self.lower_alpha_core_rad:
            raise ValueError("lower alpha core must be inside surface support")
        if not self.upper_alpha_core_rad < surface.maximum_alpha_rad:
            raise ValueError("upper alpha core must be inside surface support")
        if not surface.minimum_mach < self.lower_mach_core:
            raise ValueError("lower Mach core must be inside surface support")
        if not self.upper_mach_core < surface.maximum_mach:
            raise ValueError("upper Mach core must be inside surface support")
        if not 0.0 < self.maximum_surface_weight <= 1.0:
            raise ValueError("maximum_surface_weight must be in (0, 1]")

    @staticmethod
    def _smoothstep(value: float) -> float:
        clipped = float(np.clip(value, 0.0, 1.0))
        return clipped * clipped * (3.0 - 2.0 * clipped)

    def surface_weight_scalar(self, angle_of_attack_rad: float, mach: float) -> float:
        surface = self.surface_model.surface
        if (
            angle_of_attack_rad <= surface.minimum_alpha_rad
            or angle_of_attack_rad >= surface.maximum_alpha_rad
            or mach <= surface.minimum_mach
            or mach >= surface.maximum_mach
        ):
            return 0.0
        lower_alpha = self._smoothstep(
            (angle_of_attack_rad - surface.minimum_alpha_rad)
            / (self.lower_alpha_core_rad - surface.minimum_alpha_rad)
        )
        upper_alpha = self._smoothstep(
            (surface.maximum_alpha_rad - angle_of_attack_rad)
            / (surface.maximum_alpha_rad - self.upper_alpha_core_rad)
        )
        lower_mach = self._smoothstep(
            (mach - surface.minimum_mach)
            / (self.lower_mach_core - surface.minimum_mach)
        )
        upper_mach = self._smoothstep(
            (surface.maximum_mach - mach)
            / (surface.maximum_mach - self.upper_mach_core)
        )
        return self.maximum_surface_weight * min(
            lower_alpha, upper_alpha, lower_mach, upper_mach
        )

    def angle_and_lift_coefficient_scalar(
        self, *, pitch_angle_rad: float, flight_path_angle_rad: float, mach: float
    ) -> tuple[float, float]:
        surface_angle = self.surface_model.effective_angle_of_attack_scalar(
            pitch_angle_rad=pitch_angle_rad,
            flight_path_angle_rad=flight_path_angle_rad,
            mach=mach,
        )
        weight = self.surface_weight_scalar(surface_angle, mach)
        empirical_angle = pitch_angle_rad - flight_path_angle_rad
        empirical = self.empirical_fallback.lift_coefficient_scalar(
            empirical_angle, mach
        )
        if weight <= 0.0:
            return surface_angle, empirical
        surface = self.surface_model.surface.lift_coefficient_scalar(
            surface_angle, mach
        )
        return surface_angle, weight * surface + (1.0 - weight) * empirical
