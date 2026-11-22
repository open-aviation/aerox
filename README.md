# Aerox

Aerox is an open aircraft performance library and the successor to OpenAP. It
gives you aircraft data and performance models for 144 aircraft types, built
from open sources.

```bash
pip install aerox
```

Or, from a copy of the repository:

```bash
pip install .
```

## Quick start

Load an aircraft:

```python
from aerox import load_aircraft, get_available_aircraft

ac = load_aircraft("A320")
print(ac.mtow, ac.wing.area, ac.wing.span)

print(get_available_aircraft())
```

Get a drag polar, `CD = CD0 + k CL²`:

```python
from aerox.models.drag import build_reference_drag_polar

polar = build_reference_drag_polar(ac)
print(polar.cd0, polar.k)
print(polar.drag_coefficient(cl=0.55))   # or polar.cd(cl=0.55)
```

Get drag at a flight condition, with the Mach and Reynolds number effects:

```python
from aerox import aero
from aerox.models.drag import build_drag_model, build_drag_report

model = build_drag_model(ac)
reynolds = aero.reynolds_per_m(mach=0.78, h=11_000)   # per metre
print(model.cd(cl=0.55, mach=0.78, reynolds_per_m=reynolds))

# The contribution of each aircraft part to the zero-lift drag
for term in build_drag_report(ac).parasite_terms:
    print(term)
```

Atmosphere and speed conversions, in SI units:

```python
from aerox import aero

p, rho, T = aero.atmos(h=10_000)           # altitude in m
cas = aero.mach2cas(mach=0.78, h=10_000)   # m/s
```

## What is included

- **Aircraft data.** One YAML record per type with geometry, mass limits, and
  engine data. Each value has a source note under
  `src/aerox/data/aircraft/sources`.
- **Drag.** A clean-configuration drag model that is computed from the aircraft
  geometry: skin friction and form factors for the wing, fuselage, tails,
  nacelles, and pylons, plus wave drag and induced drag. No value is fitted to
  another performance model.
- **Atmosphere and speeds.** `aerox.aero`, ISA atmosphere and
  CAS/TAS/EAS/Mach conversions.
- **Flight management.** `aerox.fms`, route, altitude and speed intent, and
  guidance primitives.

## Accuracy and limits

- The drag model agrees with a reference performance model within 5 % at
  cruise for most jet transports. At lower Mach numbers (about 0.6) it is 4 to
  12 % too high.
- Drag is for the clean configuration only. Flap, slat, and landing gear
  increments are not included yet.
- The modules for thrust, lift, flight dynamics, and simulation are
  experimental. Their interfaces can change.

## Documentation

See [`docs/`](docs/README.md) for the aircraft data, the drag model, the
atmosphere functions, and the roadmap.

## Development

```bash
uv run pytest -q
uv run --extra editor tools/web_editor.py   # local editor for aircraft records
```

## Licence

GNU Lesser General Public License v3. See `LICENSE`.
