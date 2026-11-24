# Roadmap

Updated: 2026-11-24

## Release 0.4

- **Engine names.** Engine names are bare designations, for example
  `CFM56-5B4` and `PW1524G`, so that they match engine tables. Code that
  uses the old names must use the new names.
- **Verified limits.** Masses, VMO, MMO, and ceilings are checked against
  the FAA data sheets where the EASA data sheet refers to the flight manual.
  The C650 and MD11 limits and the masses of six piston aircraft are
  corrected. No record has a landing or
  zero-fuel mass above its takeoff mass.
- **Corrections.** The F15 and F16 engines are swapped to the correct
  types. The C310 default engine is the IO-520-M. The F2TH wing span is the
  certified 19.238 m.
- **Sources.** The notes cite the newest issues of the EASA data sheets
  A.064, A.008, IM.A.070, and IM.A.570.
- **Geometry tool.** `tools/update_aircraft_geometry.py` keeps consistent
  derived values and their notes.

## Release 0.3

- **Aircraft data.** Six types are added: A124, A310, AT76, B735, E290, and
  SU95.
- **Verified limits.** MTOW, MLW, VMO, MMO, and ceiling of the turbofan
  records are checked against type certificate data sheets. Where the
  certificate gives options, the record uses the value common in service.
- **Corrections.** The GLF6 fuel capacity is in litres, as in all other
  records. BCS1, BCS3, B38M, and CRJ7 name engine variants that exist in the
  engine table.

## Release 0.2.0

- **Flaps and landing gear.** Drag increments for deployed flaps and landing
  gear. See [Drag](drag.md#flaps-and-landing-gear).
- **JAX and CasADi.** The full drag model as a closed-form expression for
  optimization and automatic differentiation. See
  [Math backends](backends.md).
- **Aircraft data.** The Gulfstream G650 (GLF6) is added.

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
- Flap and landing gear drag use simple handbook methods, checked only against
  A320 gear drag from flight data. Slats add no drag, and spoiler drag is not
  modelled.
- Some wing reference areas have no confirmed source: the 737 MAX family, the
  787-10, the 747-8, the A330-900, the A350-1000, and the CRJ200.
- The VMO and MMO of the A359, A35K, A388, B37M, B3XM, GLF5, and LJ60, the
  CL60 limits, and the limits of the military types are not yet verified.
- The A124 has no public type certificate, so most of its values are
  inferred. The AT76 speed limits and ceiling come from training material.

## Planned work

1. **Drag increase with Mach below the drag rise.** Measured drag at a fixed
   lift coefficient rises 8 to 12 % from Mach 0.6 to 0.8. This term will be
   added together with a lower base drag, so that cruise drag stays correct
   and drag at lower Mach numbers improves.
2. **Wing reference areas.** Confirmed sources for the open values listed
   above.
3. **Flap data.** Sourced flap chord and span ratios for the records that
   use the default values, and a check of flap drag against flight data.
4. **One aircraft data schema.** One declarative schema for the aircraft
   records, their validation, and the web editor.
5. **Batched performance evaluation.** Evaluation of the atmosphere and the
   forces for many flight states in one call.
