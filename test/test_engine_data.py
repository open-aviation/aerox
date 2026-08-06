from __future__ import annotations

import csv
from pathlib import Path


ENGINE_ROOT = Path("src/aerox/data/engines")


def test_icao_edb_issue_32_and_generated_query_table_are_bundled():
    assert (ENGINE_ROOT / "icao-edb-issue-32.xlsx").stat().st_size > 500_000
    assert (ENGINE_ROOT / "generate_engine_data.py").is_file()
    assert (ENGINE_ROOT / "engine_cruise_performance.csv").is_file()
    assert (ENGINE_ROOT / "SOURCES.txt").is_file()

    with (ENGINE_ROOT / "engines.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = [row["name"] for row in rows]
    assert len(rows) == 437
    assert len(set(names)) == len(names)
    assert "CFM56-5-A1" in names
    assert "CFM56-5B4" in names
    assert "CFM56-5B4/2" in names
    assert "LEAP-1A26/26E1" in names
