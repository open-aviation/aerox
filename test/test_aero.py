import numpy as np

from aerox import aero


def test_openap_reference_atmosphere_values_are_preserved():
    pressure, density, temperature = aero.atmos(np.array([0.0, 10_000.0, 15_000.0]))

    np.testing.assert_allclose(temperature, [288.15, 223.15, 216.65])
    np.testing.assert_allclose(pressure, [101325.0, 26429.70, 12041.15], rtol=1e-4)
    np.testing.assert_allclose(density[:2], [1.225, 0.412604], rtol=1e-4)


def test_openap_cas_tas_and_mach_round_trips_are_preserved():
    cas = np.array([100.0, 150.0, 200.0])
    altitude = np.array([0.0, 5_000.0, 10_000.0])

    tas = aero.cas2tas(cas, altitude)
    np.testing.assert_allclose(aero.tas2cas(tas, altitude), cas, rtol=1e-12)
    mach = aero.cas2mach(cas, altitude)
    np.testing.assert_allclose(aero.mach2cas(mach, altitude), cas, rtol=1e-12)


def test_openap_crossover_reference_is_preserved():
    assert np.isclose(aero.crossover_alt(150.0, 0.78), 9335.23, rtol=0.01)
