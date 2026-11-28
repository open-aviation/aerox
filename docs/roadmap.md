# Roadmap

Updated: 2026-11-28

## Release 0.7

- **Airfoil classes.** The record field `wing.airfoil_class` (conventional,
  peaky, early supercritical, supercritical) sets the Korn technology factor
  of the wave drag model. The factor changes for 14 records: the B744,
  MD11, and B712 (peaky, 0.89), the FA20 (conventional, 0.87), and the A306,
  A310, 767, 757, Hawker 800, Beechjet 400, and Citation III (early
  supercritical, 0.93). Records without a class keep 0.95. See
  [Drag](drag.md#model).
- **Wing geometry.** The A350-900 and A350-1000 thickness ratio changes from
  0.123 to 0.114. The A330 and A340-300 sweep changes to 29.7° and their
  thickness ratio to 0.118. The 747-400 and 747-8 sweep changes from 37° to
  37.5°.
- **Source notes.** The geometry tool gives a separate correction reason
  for inferred values.

## Release 0.6

- **High-lift drag.** Geometry models use GASP flap profile and lift
  corrections and a Torenbeek slat term. Aircraft records include sourced
  surface geometry and settings where available. See
  [Drag](drag.md#flaps-slats-and-landing-gear).
- **Wing thickness and sweep.** Sourced wing thickness ratios or sweep
  angles for 19 records: the A380, the 777 and 757 families, the 737 Next
  Generation and MAX families, the ERJ-135 and ERJ-145, the Citation III,
  the Hawker 800, and the Falcon 20. The Citation III sweep changes from 35°
  to 23° and the Hawker 800 sweep from 15° to 20°. The clean drag of these
  records changes, by up to 7 % for the Citation III at its reference
  condition and by less than 2 % for the airliners.
- **Data view.** `aerox view` lists every field of a record with its status,
  source note, and source documents, and shows the empty fields. The web
  editor and the `editor` extra are removed.

## Release 0.5

- **Drag reference altitude.** The reference altitude of the drag model
  does not exceed the published ceiling. This changes the drag model of 42
  records, mostly business jets.
- **Verified limits.** The CL60, LJ31, LJ60, B37M, and B3XM limits are
  checked against the FAA data sheets. The LJ31 record is the Learjet 31A
  with TFE731-2-3B engines. The CL60 record is the Challenger 604 with
  CF34-3B engines.
- **Zero-fuel masses.** MZFW is added for the B37M, B3XM, C750, CL60, GLF4,
  GLF5, and LJ31.
- **Sources.** The 737 records cite FAA data sheet A16WE Revision 83.

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
  gear. See [Drag](drag.md#flaps-slats-and-landing-gear).
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
  jet transports. In climb and descent near Mach 0.6, at the flown altitude,
  it is 2 to 3 % above that model.
- Between the cruise Mach number and MMO, drag is too low: about 10 % on
  average and about 18 % at MMO, against published flight-test curves. The
  wave drag starts too late and rises too slowly.
- High-lift drag uses representative flap and slat angles. Chord, slat
  geometry and flap stations are sparse. The model has no separate terms
  for trim, slot-gap scheduling, ground effect or gear interference.
  Spoiler drag is outside the model.
- Some wing reference areas have no confirmed source: the 737 MAX family, the
  787-10, the 747-8, the A330-900, the A350-1000, and the CRJ200.
- The VMO and MMO of the A359, A35K, A388, and GLF5 and the limits of the
  military types are not yet verified. For the A388 and GLF5, the data
  sheets refer to the flight manual.
- The A124 has no public type certificate, so most of its values are
  inferred. The AT76 speed limits and ceiling come from training material.

## Planned work

1. **Drag rise above the cruise Mach number.** A wave drag method that
   starts earlier and rises faster, and sourced wing thickness ratios and
   sweep angles for the records that use default values. Two studies of
   published methods found no replacement that improves the drag near MMO
   and keeps the cruise drag. The two references used differ by about 5 %
   at cruise, so the next check uses measured flight data.
2. **Wing reference areas.** Confirmed sources for the open values listed
   above.
3. **High-lift data.** Obtain flap chord, flap and slat stations, and actual
   deflections for more aircraft. Check the model against measured
   configuration polars over a range of lift coefficients.
4. **One aircraft data schema.** One declarative schema for the aircraft
   records and their validation.
5. **Batched performance evaluation.** Evaluation of the atmosphere and the
   forces for many flight states in one call.
