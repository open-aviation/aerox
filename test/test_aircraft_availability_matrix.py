from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

from tools.generate_aircraft_availability_matrix import (
    DEFAULT_OUTPUT,
    SUMMARY_COLUMNS,
    parameter_paths,
)


def _rows() -> list[dict[str, str]]:
    with DEFAULT_OUTPUT.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def test_aircraft_availability_matrix_is_current():
    result = subprocess.run(
        [
            sys.executable,
            "tools/generate_aircraft_availability_matrix.py",
            "--check",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_aircraft_availability_matrix_has_all_aircraft_and_parameters():
    with DEFAULT_OUTPUT.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [*SUMMARY_COLUMNS, *parameter_paths()]
        rows = list(reader)

    assert len(rows) == 144
    assert len({row["icao"] for row in rows}) == 144
    assert [row["icao"] for row in rows] == sorted(row["icao"] for row in rows)

    for row in rows:
        flags = [row[path] for path in parameter_paths()]
        assert set(flags) <= {"0", "1"}
        assert int(row["available_parameter_count"]) == flags.count("1")
        assert int(row["total_parameter_count"]) == len(parameter_paths())
        expected_percent = 100 * flags.count("1") / len(flags)
        assert float(row["availability_percent"]) == round(expected_percent, 2)
        assert Path(row["yaml_file"]).is_file()


def test_matrix_exposes_known_available_and_unavailable_tail_fields():
    rows = {row["icao"]: row for row in _rows()}

    assert rows["A320"]["horizontal_tail.area"] == "1"
    assert rows["A320"]["vertical_tail.area"] == "1"
    assert rows["PC24"]["horizontal_tail.span"] == "1"
    assert rows["PC24"]["horizontal_tail.area"] == "0"
    assert rows["B737"]["horizontal_tail.span"] == "1"
    assert rows["B737"]["horizontal_tail.area"] == "0"
    assert rows["A320"]["subtypes"] == "1"
    assert rows["B38M"]["subtypes"] == "0"
    assert rows["B37M"]["subtypes"] == "0"
