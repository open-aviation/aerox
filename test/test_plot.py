import numpy as np
import pytest

from aerox import get_available_aircraft, load_aircraft
from aerox.plot import aircraft_geometry


def test_geometry_builds_for_every_aircraft():
    for code in get_available_aircraft():
        geometry = aircraft_geometry(load_aircraft(code))
        names = {part.name for part in geometry.parts}
        assert {"fuselage", "wing", "horizontal_tail", "vertical_tail"} <= names
        points = geometry.points()
        assert np.all(np.isfinite(points)), code


def test_geometry_matches_the_record_span_and_length():
    aircraft = load_aircraft("A320")
    points = aircraft_geometry(aircraft).points()
    assert np.ptp(points[:, 1]) == pytest.approx(aircraft.wing.span, rel=0.01)
    assert points[:, 0].max() == pytest.approx(aircraft.fuselage.length, rel=0.01)


def test_three_view_and_3d_view_draw():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from aerox.plot import three_view, view_3d

    for code in ("A320", "AT72", "C172"):
        aircraft = load_aircraft(code)
        plt.close(three_view(aircraft))
        plt.close(view_3d(aircraft))


def test_write_viewer_embeds_the_selected_aircraft(tmp_path):
    from aerox.plot import write_viewer

    path = tmp_path / "viewer.html"
    write_viewer(path, [load_aircraft("A320"), load_aircraft("AT72")])
    html = path.read_text(encoding="utf-8")
    assert "__AIRCRAFT_DATA__" not in html and "__PART_COLORS__" not in html
    assert '"A320":' in html and '"AT72":' in html
    assert "three@0.160.0" in html
