import numpy as np

from aerox.models.atmosphere import (
    atmosphere_at_pressure_altitude,
    calibrated_airspeed_from_true_airspeed,
    dynamic_pressure_from_calibrated_airspeed,
    dynamic_viscosity_air,
    isa_atmosphere,
    mach_from_calibrated_airspeed,
    mach_from_true_airspeed,
    true_airspeed_from_calibrated_airspeed,
    unit_reynolds_number,
)


def test_isa_reference_states_at_sea_level_and_tropopause():
    state = isa_atmosphere(np.array([0.0, 11_000.0]))

    # These are the exact OpenAP 2.6.0 implementation values.  Its rounded
    # sea-level density and fitted exponent differ slightly from table ISA.
    np.testing.assert_allclose(
        state.pressure_pa, [101_324.9985, 22_625.7912], rtol=2e-6
    )
    np.testing.assert_allclose(state.temperature_k, [288.15, 216.65], atol=1e-12)
    np.testing.assert_allclose(
        state.density_kg_m3, [1.225, 0.36381717], rtol=2e-5
    )


def test_supplied_temperature_changes_density_but_not_pressure():
    standard = isa_atmosphere(np.array([5_000.0]))
    warm = atmosphere_at_pressure_altitude(np.array([5_000.0]), np.array([280.0]))

    np.testing.assert_allclose(warm.pressure_pa, standard.pressure_pa)
    assert warm.density_kg_m3[0] < standard.density_kg_m3[0]


def test_sutherland_viscosity_and_unit_reynolds_number_are_vectorized():
    temperature = np.array([288.15, 216.65])
    viscosity = dynamic_viscosity_air(temperature)

    np.testing.assert_allclose(viscosity[0], 1.78938e-5, rtol=2e-5)
    assert viscosity[1] < viscosity[0]
    np.testing.assert_allclose(
        unit_reynolds_number(np.array([1.225, 0.36]), 200.0, temperature),
        np.array([1.225, 0.36]) * 200.0 / viscosity,
    )


def test_cas_air_data_round_trip_at_sea_level():
    cas = np.array([100.0, 200.0])
    altitude = np.zeros(2)
    state = isa_atmosphere(np.zeros(2))

    mach = mach_from_calibrated_airspeed(cas, altitude)
    dynamic_pressure = dynamic_pressure_from_calibrated_airspeed(cas, altitude)

    np.testing.assert_allclose(mach, cas / state.speed_of_sound_mps, rtol=1e-8)
    np.testing.assert_allclose(
        dynamic_pressure, 0.5 * state.density_kg_m3 * cas**2, rtol=1e-8
    )


def test_pressure_altitude_airdata_round_trip_with_non_isa_temperature():
    cas = np.array([140.0, 150.0])
    altitude = np.array([5_000.0, 11_000.0])
    temperature = isa_atmosphere(altitude).temperature_k + 12.0

    tas = true_airspeed_from_calibrated_airspeed(cas, altitude, temperature)
    recovered_cas = calibrated_airspeed_from_true_airspeed(
        tas, altitude, temperature
    )
    mach = mach_from_true_airspeed(tas, altitude, temperature)

    np.testing.assert_allclose(recovered_cas, cas, rtol=1e-12, atol=1e-10)
    np.testing.assert_allclose(
        mach, mach_from_calibrated_airspeed(cas, altitude), rtol=1e-12
    )
