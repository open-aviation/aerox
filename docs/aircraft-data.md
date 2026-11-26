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
| Flaps and slats | `flaps`: type, area, stations, area and chord ratios, named settings | m², deg |
| Engines | `engine`: `type`, `mount`, `number`, `default`, `options` | |
| Drag | `drag`: the drag model and its reference condition, see [Drag](drag.md) | |
| Kinematics | `kinematic`: takeoff distance and V2, climb speeds and vertical rates | m, kt, ft/min |

The wing thickness ratio is stored as `t/c` in YAML and as `wing.t_c` in
Python.

### Flaps and slats

`flaps.type` accepts `plain`, `split`, `single-slotted`, `double-slotted`,
`triple-slotted`, `fowler`, `double-slotted-fowler` and
`mixed-single-double-slotted`. The last value describes aircraft with
separate single- and double-slotted surfaces.

`eta_inboard` and `eta_outboard` are station distances measured from the
aircraft centreline and divided by the wing semi-span. `flapped_area_ratio`
and `slatted_area_ratio` refer to the wing planform served by the surfaces,
divided by the full reference area. `chord_ratio` is the representative flap
chord divided by the local clean wing chord. `area` is the reported total
trailing-edge flap surface area on both sides. The source note states the
available definition of this area. The model uses the served wing area ratio
directly or derives it from the flap stations.

The old fields `bf/b`, `lambda_f`, `cf/c` and `Sf/S` are removed. For
custom records, replace them with sourced fields that match the definitions
above or give `null` to use the model defaults.

`settings` maps case-sensitive names to `flap_angle_deg` and optional
`slat_angle_deg`. Quote numeric names in YAML. Use `slats_extended` when a
source reports the slat state without an angle. When the source lists
different angles across the span, the records use a representative angle
and give the original values in the source note.

```python
flap, slat = load_aircraft("A320").flaps.angles_rad("FULL-V2500")
```

`angles_rad` returns the two angles in radians. The slat angle is `None`
when the record gives no value. A missing setting raises `KeyError`. See
[Drag](drag.md#flaps-slats-and-landing-gear) for class defaults and the
configuration inputs.

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

## View and edit records

The viewer shows every value of a record with its status, source, and
notes. Select **Data** in the page, or press `T`:

```bash
aerox view A320
```

The Data view is read-only. It shows the review date and certification
scope of the source note and a bar of the values by status. Filter the
fields by text, status, or source document. Select **Empty fields** to also
list the fields the record does not fill. Select a field to see its full
source note and to copy its path or value. The cited documents are listed
at the end with their titles, dates, and links. A link such as
`#A320/data` at the end of the page address opens an aircraft in a view.
The Data view also works without an internet connection or WebGL.

To change a value, edit the record and its source note, then apply the
tools in `tools/` (`update_aircraft_geometry.py`,
`update_aircraft_drag_models.py`, and
`generate_aircraft_availability_matrix.py`). Run the tests after an edit:

```bash
uv run pytest -q
```
