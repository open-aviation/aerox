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
    for key in (
        "__AIRCRAFT_DATA__",
        "__PART_COLORS__",
        "__RECORD_TEXT__",
        "__SOURCES__",
    ):
        assert key not in html
    assert '"A320":' in html and '"AT72":' in html
    assert "three@0.160.0" in html


def test_viewer_record_rows_carry_values_and_source_notes():
    from aerox.plot import _record_rows

    rows = {row[0]: row for row in _record_rows(load_aircraft("A320"))}
    assert rows["mtow"][1:5] == [
        78000,
        "kg",
        "certified-published",
        "easa_a064_issue_63",
    ]
    assert rows["wing.t/c"][1] > 0
    assert rows["flaps.settings.FULL.flap_angle_deg"][1] == 35
    assert rows["subtypes.A320-214"][1] == "CFM56-5B4, CFM56-5B4/2"
    assert rows["kinematic.cruise_mach"][3] == ""
    assert not any(path.startswith("data_provenance") for path in rows)
    assert not any(path.endswith(".source_id") for path in rows)
    # Details of the source note, and empty fields with their search record
    assert rows["wing.span"][5]["page_or_section"] == "A320 principal dimensions"
    assert rows["wing.tip_chord"][1] is None
    assert rows["wing.tip_chord"][3] == "unresolved"
    assert "reason_unavailable" in rows["wing.tip_chord"][5]
    assert rows["aileron.chord_ratio"][1:5] == [None, "", "", ""]


def test_viewer_header_and_source_documents():
    from aerox.plot import _record_header, _source_documents

    aircraft = load_aircraft("A320")
    header = _record_header(aircraft)
    assert header["reviewed"] == "2026-11-24"
    assert ["Type certificate", "EASA.A.064"] in header["scope"]
    documents = _source_documents([aircraft])
    title, _, url = documents["easa_a064_issue_63"]
    assert "EASA.A.064" in title and url.startswith("https://")


def test_write_viewer_keeps_each_note_text_once(tmp_path):
    import json
    import re

    from aerox.plot import write_viewer

    path = tmp_path / "viewer.html"
    write_viewer(path, [load_aircraft("A320"), load_aircraft("A321")])
    html = path.read_text(encoding="utf-8")
    text = json.loads(re.search(r"const TEXT = (.*);", html).group(1))
    assert len(text) == len(set(text))
    assert "page_or_section" in text
