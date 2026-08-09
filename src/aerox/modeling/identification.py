"""Small, dependency-light identification primitives for grey-box models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class RobustLinearFit:
    """Result of a weighted Huber iteratively reweighted least-squares fit."""

    coefficients: FloatArray
    residuals: FloatArray
    base_weighted_rmse: float
    huber_objective: float
    robust_scale: float
    normal_matrix_condition: float
    coefficient_correlation: FloatArray
    iterations: int


def delay_samples(values: FloatArray, samples: int) -> FloatArray:
    """Return a causal delay where output[t] = input[t - samples]."""
    data = np.asarray(values, dtype=float)
    if data.ndim != 1:
        raise ValueError("values must be one-dimensional")
    if samples < 0:
        raise ValueError("samples must be non-negative for a causal delay")
    output = np.full_like(data, np.nan, dtype=float)
    if samples == 0:
        output[:] = data
    elif samples < len(data):
        output[samples:] = data[:-samples]
    return output


def weighted_rmse(residuals: FloatArray, weights: FloatArray) -> float:
    residual = np.asarray(residuals, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if residual.shape != weight.shape or residual.ndim != 1:
        raise ValueError("residuals and weights must be equal-length vectors")
    if np.any(weight < 0) or not np.any(weight > 0):
        raise ValueError(
            "weights must be non-negative with at least one positive value"
        )
    return float(np.sqrt(np.average(residual**2, weights=weight)))


def robust_weighted_linear_fit(
    design: FloatArray,
    target: FloatArray,
    weights: FloatArray,
    *,
    huber_delta: float = 1.5,
    max_iterations: int = 50,
    tolerance: float = 1e-10,
) -> RobustLinearFit:
    """Fit a linear model with base weights and Huber robust reweighting."""
    x = np.asarray(design, dtype=float)
    y = np.asarray(target, dtype=float)
    base_weights = np.asarray(weights, dtype=float)
    if x.ndim != 2 or y.ndim != 1 or base_weights.ndim != 1:
        raise ValueError("design must be 2-D; target and weights must be 1-D")
    if len(x) != len(y) or len(y) != len(base_weights):
        raise ValueError("design, target, and weights must have equal row counts")
    if np.any(~np.isfinite(x)) or np.any(~np.isfinite(y)):
        raise ValueError("design and target must be finite")
    if np.any(base_weights < 0) or not np.any(base_weights > 0):
        raise ValueError(
            "weights must be non-negative with at least one positive value"
        )
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")

    robust_weights = np.ones(len(y), dtype=float)
    coefficients = np.zeros(x.shape[1], dtype=float)
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        combined = base_weights * robust_weights
        square_root = np.sqrt(combined)
        weighted_design = x * square_root[:, None]
        weighted_target = y * square_root
        updated = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
        residuals = y - x @ updated
        centered = residuals - np.median(residuals)
        scale = 1.4826 * np.median(np.abs(centered)) + 1e-12
        cutoff = huber_delta * scale
        robust_weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residuals), 1e-15))
        if np.max(np.abs(updated - coefficients)) <= tolerance:
            coefficients = updated
            break
        coefficients = updated

    residuals = y - x @ coefficients
    absolute = np.abs(residuals)
    huber_loss = np.where(
        absolute <= cutoff,
        0.5 * residuals**2,
        cutoff * (absolute - 0.5 * cutoff),
    )
    combined = base_weights * robust_weights
    weighted_design = x * np.sqrt(combined)[:, None]
    normal = weighted_design.T @ weighted_design
    covariance_shape = np.linalg.pinv(normal)
    diagonal = np.sqrt(np.maximum(np.diag(covariance_shape), 0.0))
    denominator = np.outer(diagonal, diagonal)
    correlation = np.divide(
        covariance_shape,
        denominator,
        out=np.zeros_like(covariance_shape),
        where=denominator > 0,
    )
    return RobustLinearFit(
        coefficients=coefficients,
        residuals=residuals,
        base_weighted_rmse=weighted_rmse(residuals, base_weights),
        huber_objective=float(np.average(huber_loss, weights=base_weights)),
        robust_scale=float(scale),
        normal_matrix_condition=float(np.linalg.cond(normal)),
        coefficient_correlation=correlation,
        iterations=iterations,
    )
