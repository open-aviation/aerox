"""Forward-only A320 configuration decoding for complete-flight replay."""

from __future__ import annotations

from collections.abc import Sequence
from enum import IntEnum

import numpy as np

from .a320_qar import (
    ChannelSpec,
    DecodeStatus,
    FloatArray,
    bounded_sample_and_hold,
)


class A320FlapConfiguration(IntEnum):
    """Provisional ordinal interpretation of the A320 flap-handle detents."""

    CLEAN = 0
    CONF_1 = 1
    CONF_2 = 2
    CONF_3 = 3
    FULL = 4


# FLAP_RAW is an angular recorder encoding rather than a surface angle. These
# five modes account for more than 99% of native samples in both raw aircraft
# files. Their ordinal labels follow the configuration sequence observed on the
# development/calibration flights; they do not assert slat/flap surface angles.
A320_FLAP_RAW_DETENTS: dict[A320FlapConfiguration, float] = {
    A320FlapConfiguration.CLEAN: 0.0,
    A320FlapConfiguration.CONF_1: 119.53125,
    A320FlapConfiguration.CONF_2: 144.84375,
    A320FlapConfiguration.CONF_3: 167.34375,
    A320FlapConfiguration.FULL: -129.375,
}
A320_FLAP_RAW_QUANTUM = 1.40625
A320_FLAP_CONFIGURATION_SPEC = ChannelSpec(
    logical_name="flap_configuration",
    raw_name="FLAP_RAW",
    raw_unit="recorder_code",
    output_unit="ordinal_configuration",
    status=DecodeStatus.PROVISIONAL,
    max_hold_s=2.0,
)


def decode_a320_flap_configuration(
    raw: Sequence[float],
    time_s: Sequence[float],
    *,
    max_hold_s: float = 2.0,
    tolerance: float = A320_FLAP_RAW_QUANTUM + 1e-9,
) -> FloatArray:
    """Decode canonical ``FLAP_RAW`` modes to provisional ordinal detents.

    Values farther than one recorder quantum from a canonical mode are treated
    as transition or invalid samples. A bounded hold bridges only the native
    two-second sampling interval; it never turns a long outage into a known
    configuration.
    """
    values = np.asarray(raw, dtype=float)
    time = np.asarray(time_s, dtype=float)
    if values.ndim != 1 or time.ndim != 1 or len(values) != len(time):
        raise ValueError("raw and time_s must be equal-length one-dimensional arrays")
    if tolerance < 0.0:
        raise ValueError("tolerance must be non-negative")
    configurations = np.asarray(list(A320_FLAP_RAW_DETENTS), dtype=float)
    detents = np.asarray(list(A320_FLAP_RAW_DETENTS.values()), dtype=float)
    distance = np.abs(values[:, np.newaxis] - detents[np.newaxis, :])
    nearest = np.argmin(distance, axis=1)
    decoded = np.full(len(values), np.nan, dtype=float)
    accepted = np.isfinite(values) & (
        distance[np.arange(len(values)), nearest] <= tolerance
    )
    decoded[accepted] = configurations[nearest[accepted]]
    return bounded_sample_and_hold(decoded, time, max_hold_s)
