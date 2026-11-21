from datetime import datetime

import numpy as np
import pytest

from aerox.flightdata.a320_configuration import (
    A320_FLAP_RAW_DETENTS,
    A320FlapConfiguration,
    decode_a320_flap_configuration,
)
from aerox.flightdata.a320_qar import (
    A320_CHANNELS,
    DecodeStatus,
    DecodedA320Flight,
    DecodedChannel,
    bounded_sample_and_hold,
    reconstruct_utc_timestamps,
)


def test_bounded_hold_respects_gap_and_small_initial_backfill():
    time = np.arange(7.0)
    values = np.array([np.nan, 10.0, np.nan, np.nan, np.nan, 20.0, np.nan])

    held = bounded_sample_and_hold(values, time, 2.0)

    np.testing.assert_allclose(
        held, [10.0, 10.0, 10.0, 10.0, np.nan, 20.0, 20.0], equal_nan=True
    )


def test_timestamp_reconstruction_handles_staggered_date_and_utc_samples():
    timestamps = reconstruct_utc_timestamps(
        [0, 1000, 2000, 3000, 4000, 5000],
        ["", "01/08", "", "", "", "01/08"],
        ["", "", "12:00:02", "", "", "12:00:05"],
        year=2011,
    )

    assert timestamps[0] == np.datetime64(datetime(2011, 8, 1, 12, 0, 0), "ns")
    assert timestamps[-1] == np.datetime64(datetime(2011, 8, 1, 12, 0, 5), "ns")


def test_provisional_channels_require_explicit_opt_in():
    spec = A320_CHANNELS["pitch"]
    flight = DecodedA320Flight(
        registration="TAIL-A",
        recording="REC1",
        time_s=np.array([0.0]),
        timestamp_utc=np.array([np.datetime64("2011-08-01T12:00:00", "ns")]),
        airborne=np.array([True]),
        channels={"pitch": DecodedChannel(np.array([0.1]), spec)},
    )

    assert spec.status is DecodeStatus.PROVISIONAL
    with pytest.raises(ValueError, match="provisional conversion"):
        flight.channel("pitch")
    np.testing.assert_array_equal(
        flight.channel("pitch", allow_provisional=True), [0.1]
    )


def test_flap_configuration_decoder_maps_detents_and_one_quantum_jitter():
    time = np.arange(7.0)
    raw = np.array(
        [
            A320_FLAP_RAW_DETENTS[A320FlapConfiguration.CLEAN],
            np.nan,
            A320_FLAP_RAW_DETENTS[A320FlapConfiguration.CONF_1] + 1.40625,
            140.625,
            np.nan,
            np.nan,
            np.nan,
        ]
    )

    decoded = decode_a320_flap_configuration(raw, time)

    np.testing.assert_allclose(
        decoded,
        [
            A320FlapConfiguration.CLEAN,
            A320FlapConfiguration.CLEAN,
            A320FlapConfiguration.CONF_1,
            A320FlapConfiguration.CONF_1,
            A320FlapConfiguration.CONF_1,
            np.nan,
            np.nan,
        ],
        equal_nan=True,
    )


def test_flap_configuration_decoder_rejects_invalid_shapes_and_tolerance():
    with pytest.raises(ValueError, match="equal-length"):
        decode_a320_flap_configuration([0.0], [0.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        decode_a320_flap_configuration([0.0], [0.0], tolerance=-1.0)
