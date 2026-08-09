import numpy as np

from aerox.modeling.identification import (
    delay_samples,
    robust_weighted_linear_fit,
)


def test_delay_samples_is_causal_and_marks_unavailable_prefix():
    delayed = delay_samples(np.array([1.0, 2.0, 3.0, 4.0]), 2)

    np.testing.assert_allclose(delayed, [np.nan, np.nan, 1.0, 2.0], equal_nan=True)


def test_weighted_huber_fit_limits_a_large_outlier():
    x = np.arange(20.0)
    design = np.column_stack((x, np.ones_like(x)))
    target = 2.0 * x + 3.0
    target[-1] += 1_000.0

    fit = robust_weighted_linear_fit(design, target, np.ones_like(target))

    np.testing.assert_allclose(fit.coefficients, [2.0, 3.0], atol=1e-5)
    assert fit.iterations > 1
    assert fit.robust_scale < 1e-6
