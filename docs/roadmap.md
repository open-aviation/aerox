# Roadmap

Updated: 2026-11-22

## Release 0.1.0

- **Aircraft data.** 144 aircraft records, each with a source note. See
  [Aircraft data](aircraft-data.md).
- **Drag.** A clean-configuration model computed from the aircraft geometry,
  with Mach and Reynolds number effects and wave drag. See [Drag](drag.md).
- **Atmosphere and speeds.** ISA atmosphere and airspeed conversions. See
  [Atmosphere and speeds](atmosphere.md).
- **Experimental.** Thrust, lift, flight dynamics, simulation, and flight
  management.

## Known limits

- Drag agrees with a reference performance model within 5 % at cruise for most
  jet transports. Near Mach 0.6 it is 4 to 12 % too high.
- Drag covers the clean configuration only.
- Some wing reference areas have no confirmed source: the 737 MAX family, the
  787-10, the 747-8, the A330-900, the A350-1000, and the CRJ200.

## Planned work

1. **Drag increase with Mach below the drag rise.** Measured drag at a fixed
   lift coefficient rises 8 to 12 % from Mach 0.6 to 0.8. This term will be
   added together with a lower base drag, so that cruise drag stays correct
   and drag at lower Mach numbers improves.
2. **Deployed configuration.** Drag increments for flaps, slats, and landing
   gear.
3. **Wing reference areas.** Confirmed sources for the open values listed
   above.
4. **One aircraft data schema.** One declarative schema for the aircraft
   records, their validation, and the web editor.
5. **Batched performance evaluation.** Evaluation of the atmosphere and the
   forces for many flight states in one call.
