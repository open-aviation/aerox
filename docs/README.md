# Aerox documentation

Aerox is an open aircraft performance library. It gives you aircraft data and
performance models for 144 aircraft types, built from open sources.

## Pages

- [Aircraft data](aircraft-data.md): the aircraft records, their units, how to
  load them, and where each value comes from.
- [Drag](drag.md): the geometry-based drag model, how to evaluate it, and its
  accuracy.
- [Atmosphere and speeds](atmosphere.md): ISA atmosphere and airspeed
  conversions in `aerox.aero`.
- [Roadmap](roadmap.md): the state of the current release, its known limits,
  and the planned work.

## Install

```bash
pip install aerox
```

The runtime needs NumPy, SciPy, and PyYAML. Two extras are available:

- `aerox[editor]` adds Flask for the local editor of aircraft records.
- `aerox[research]` adds pandas, PyArrow, and Matplotlib.

## Stability

The aircraft data, the drag model, and `aerox.aero` are the supported parts of
the library. The modules for thrust, lift, flight dynamics, simulation, and
flight management (`aerox.fms`) are experimental. Their interfaces can change
between releases.
