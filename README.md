# Aerox

Aerox is a new aircraft performance data library and the successor project to
OpenAP. The first version is intentionally small: it packages curated aircraft
YAML files and exposes typed Python helpers for loading them.

## Current Scope

- `aerox.aircraft` dataclasses for aircraft configuration data
- bundled aircraft YAML data under `src/aerox/data/aircraft/yml`
- a local Flask editor for maintaining the aircraft YAML files

OpenAP-derived performance models such as fuel, emissions, mass, drag, thrust,
trajectory generation, BADA, and CasADi are not part of the initial runtime
package.

## Usage

```python
from aerox import load_aircraft

aircraft = load_aircraft("AT72")
print(aircraft.mtow)
```

## Data Editor

From the repository root:

```bash
uv run tools/web_editor.py
```

Then open `http://localhost:5000`.

## Tests

```bash
uv run pytest -q
```
