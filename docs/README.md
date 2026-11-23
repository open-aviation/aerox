# Aerox documentation

Aerox is an open aircraft performance library. It gives you aircraft data and
performance models for 140+ aircraft types, built from open sources.

## Pages

- [Aircraft data](aircraft-data.md): the aircraft records, their units, how to
  load them, and where each value comes from.
- [Drag](drag.md): the geometry-based drag model and the simpler parabolic
  polar, how to evaluate them, and their accuracy.
- [Math backends](backends.md): the drag model with JAX and CasADi, for
  optimization and automatic differentiation.
- [Kinematics](kinematics.md): the typical speeds and vertical rates of each
  flight phase, and how to build a simple climb or descent profile from them.
- [Atmosphere and speeds](atmosphere.md): ISA atmosphere and airspeed
  conversions in `aerox.aero`.
- [Roadmap](roadmap.md): the state of the current release, its known limits,
  and the planned work.

## Install

```bash
pip install aerox
```

Or, from a copy of the repository:

```bash
pip install .
```

The runtime needs NumPy, SciPy, and PyYAML. Three extras are available:

- `aerox[editor]` adds Flask for the local editor of aircraft records.
- `aerox[plot]` adds Matplotlib, to draw aircraft with `aerox.plot`.
- `aerox[research]` adds pandas, PyArrow, and Matplotlib.

## Stability

The aircraft data, the drag model, and `aerox.aero` are the supported parts of
the library. The modules for thrust, lift, flight dynamics, simulation, and
flight management (`aerox.fms`) are experimental, and so are the drawings
(`aerox.plot` and `aerox view`). Their interfaces can change between releases.
