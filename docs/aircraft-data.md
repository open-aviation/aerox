# Aircraft data

aerox has one YAML record per aircraft type, 150+ in total, grouped into
turbofan, turboprop, and piston aircraft.

## Load an aircraft

```python
from aerox import get_available_aircraft, load_aircraft

print(get_available_aircraft())   # ICAO type codes, for example "A320", "B738"

ac = load_aircraft("A320")
print(ac.aircraft)                # "Airbus A320-214"
print(ac.mtow, ac.oew)            # kg
print(ac.wing.area, ac.wing.span) # m², m
print(ac.engine.default)          # "CFM56-5B4"
```

`load_aircraft` takes the ICAO type code. It raises `FileNotFoundError` for an
unknown code. To load a record from your own folder, pass `data_dir`. To build
an aircraft from a dictionary, use `load_aircraft_from_dict`.

A field that has no value in the record is `None`.

## Record content

The main groups of a record, with their units:

| Group | Content | Units |
| --- | --- | --- |
| Identification | `aircraft`, `icao`, `iata`, `subtypes` | |
| Certification | `certification`: authority, type certificate, issue | |
| Masses | `mtow`, `mlw`, `mzfw`, `oew`, `minimum_weight` | kg |
| Speed limits | `vmo` (knots, indicated), `mmo` (Mach) | kt, - |
| Ceiling | `ceiling` | ft |
| Wake category | `icao_wtc`, `eu_wtc`, `faa_wtc` | |
| Passengers | `pax`: `max`, `high`, `low` | seats |
| Fuselage | `fuselage`: `length`, `height`, `width` | m |
| Wing | `wing`: `area`, `span`, `mac`, `sweep`, `t/c`, chords, `taper_ratio`, `aspect_ratio`, `dihedral` | m², m, deg |
| Tails | `horizontal_tail`, `vertical_tail`: area, span or height, chords, sweep, arm, control-surface area ratio | m², m, deg |
| Flaps | `flaps`: type, area, span and chord ratios | |
| Engines | `engine`: `type`, `mount`, `number`, `default`, `options` | |
| Drag | `drag`: the drag model and its reference condition, see [Drag](drag.md) | |
| Kinematics | `kinematic`: takeoff distance and V2, climb speeds and vertical rates | m, kt, ft/min |

The wing thickness ratio is stored as `t/c` in YAML and as `wing.t_c` in
Python.

`src/aerox/data/engines/engines.csv` is a table of turbofan engines. Most
turbofan names in `engine.options` match the `name` column of this table, but
not all. The engine names of turboprop and piston aircraft are descriptions
only and have no entry in the table.

## Sources

Every record has a source note in `src/aerox/data/aircraft/sources/`. The note
lists the source of each value, except the kinematic values, which have no
source notes yet (see [Kinematics](kinematics.md)). The cited documents are listed in
`sources/sources.yml` at the root of the repository. The CAD drawings and the
measurement tables behind the geometry values are in the same `sources/`
folder. These files are in the repository only, not in the installed
package.

`src/aerox/data/aircraft/aircraft-parameter-availability.csv` shows which
parameters each record contains.

## Edit records

A local web editor shows and edits the records:

```bash
uv run --extra editor tools/web_editor.py
```

Run the tests after an edit:

```bash
uv run pytest -q
```
