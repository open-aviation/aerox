import math
from pathlib import Path
from runpy import run_path


AUDIT = run_path(Path(__file__).parents[1] / "tools" / "audit_a320_qar.py")
FlightSummary = AUDIT["FlightSummary"]
haversine_nm = AUDIT["haversine_nm"]
mark_identifier_collisions = AUDIT["mark_identifier_collisions"]
select_diverse = AUDIT["select_diverse"]


def _summary(registration: str, recording: str, offset: float) -> FlightSummary:
    return FlightSummary(
        registration=registration,
        recording=recording,
        date="2011-08-01",
        airborne_duration_s=3_000.0 + offset,
        ground_before_liftoff=True,
        ground_after_touchdown=True,
        max_altitude_ft=30_000.0 + offset,
        max_cas_kts=280.0 + offset,
        takeoff_gross_weight_kg=62_000.0 + offset,
        endpoint_distance_nm=300.0 + offset,
        mean_wind_speed_raw=10.0 + offset,
        surface_activity_raw=5.0 + offset,
        min_native_coverage=1.0,
        max_critical_gap_s=2.0,
        operational_flight=True,
        in_processed_table=True,
    )


def test_haversine_is_about_sixty_nautical_miles_per_equatorial_degree():
    assert math.isclose(haversine_nm(0.0, 0.0, 0.0, 1.0), 60.04, abs_tol=0.01)


def test_identifier_collisions_require_registration_aware_keys():
    flights = [
        _summary("TAIL-A", "REC1", 0.0),
        _summary("TAIL-B", "REC1", 1.0),
        _summary("TAIL-B", "REC2", 2.0),
    ]

    assert mark_identifier_collisions(flights) == {"REC1"}
    assert [flight.identifier_collision for flight in flights] == [True, True, False]


def test_diverse_selection_is_deterministic_and_has_no_duplicates():
    flights = [_summary("TAIL-A", f"REC{i}", float(i)) for i in range(8)]

    first = select_diverse(flights, 4)
    second = select_diverse(list(reversed(flights)), 4)

    assert [item.recording for item in first] == [item.recording for item in second]
    assert len({item.recording for item in first}) == 4
