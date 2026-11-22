"""Serve the aircraft viewer; the same as ``aerox view``.

uv run tools/aircraft_viewer.py [ICAO ...] [--port N] [--no-open] [--out FILE]
"""

import sys

from aerox.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["view", *sys.argv[1:]]))
