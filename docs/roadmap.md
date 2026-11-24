# Roadmap

Updated: 2026-11-24

## Release 0.3.0

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
- About 57 VMO and MMO values (mostly airliners) and 22 ceilings are not yet
  verified. For these types, the EASA data sheet refers to the flight manual;
  the FAA data sheets are the next source to check.
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
