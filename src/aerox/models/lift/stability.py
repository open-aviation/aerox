"""Tabulated static-stability priors."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


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


__all__ = ["StaticAerodynamicCoefficients", "TabulatedStaticStabilityPrior"]

