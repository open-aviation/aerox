# Drag

The drag model computes the drag coefficient of the clean aircraft from its
geometry. No value is fitted to another performance model.

## Model

The drag coefficient has four parts:

```
CD = CD0(Re, M) + CDi(CL) + CDp(CL) + CDw(CL, M)
```

- **Zero-lift drag, `CD0`.** For each aircraft part, the model multiplies
  turbulent flat-plate skin friction by a form factor, an interference factor,
  and the wetted area. Skin friction depends on the Reynolds number and the
  Mach number. Which parts are included depends on the aircraft, see below.
- **Induced drag, `CDi`.** `CL² / (π A e)`, with the span efficiency factor `e`
  stored in the aircraft record.
- **Lift-dependent profile drag, `CDp`.** Grows with `CD0` and `CL²`.
- **Wave drag, `CDw`.** The Korn equation with Lock's fourth-power law. It is
  zero below the critical Mach number.

The model is identified as `aerox-geometry-drag-v2` and the wave drag as
`aerox-korn-lock-wave-v1` in the `drag` group of each record.

### Two kinds of aircraft

| | Civil turbofans (92 types) | Turboprops, pistons, military jets (52 types) |
| --- | --- | --- |
| Parts with skin friction | wing, fuselage, tails, nacelles, pylons | wing, fuselage, tails |
| Fixed increments | fuselage upsweep, control-surface gaps, nacelle base, roughness and excrescence (9 %) | one `unresolved` term, 28 to 35 % |
| Wave drag | yes, except the C-17 and KC-135 | no |

The fixed increments are computed once, at the reference condition of the
aircraft, and do not change with the Reynolds or Mach number.

## Evaluate drag at a flight condition

```python
from aerox import aero, load_aircraft
from aerox.models.drag import build_drag_model

ac = load_aircraft("A320")
model = build_drag_model(ac)

reynolds = aero.reynolds_per_m(0.78, 11_000)   # about 5.9e6 per metre
cd = model.drag_coefficient(0.55, mach=0.78, reynolds_per_m=reynolds)
```

`drag_coefficient(lift_coefficient, mach, reynolds_per_m)` takes the lift
coefficient, the Mach number, and the Reynolds number per metre. Each aircraft part applies its own reference length
to the Reynolds number per metre. All three arguments accept NumPy arrays and
broadcast against each other.

`aero.reynolds_per_m(mach, h)` gives the Reynolds number per metre in the ISA
atmosphere, with Sutherland's law for the viscosity. Without `reynolds_per_m`,
the model uses the reference value stored in the aircraft record
(`drag.reference_reynolds_per_m`).

The flight condition can also be passed as one object:

```python
from aerox.models.drag import DragCondition

condition = DragCondition(mach=0.78, reynolds_per_m=reynolds)
cd = model.drag_coefficient(0.55, condition=condition)
```

The model covers the clean configuration only. A condition with deployed
flaps, landing gear, or spoilers raises `ValueError`, and so does a negative
Mach number or Reynolds number.

## Drag breakdown

`breakdown` takes the same arguments and returns each part of the drag:

```python
b = model.breakdown(0.55, mach=0.78, reynolds_per_m=reynolds)
print(b.component_parasite)    # zero-lift drag of each aircraft part
print(b.parasite_increments)   # the fixed increments, by name
print(b.parasite)              # total zero-lift drag
print(b.span_induced, b.lift_dependent_profile, b.total)
```

For aircraft with wave drag (the A320 above), the breakdown also has
`b.wave` and `b.critical_mach`. For the other aircraft these two fields do not
exist.

`build_drag_report` rebuilds the drag model parameters from the aircraft
geometry and lists the zero-lift terms, with the method used for each term:

```python
from aerox.models.drag import build_drag_report

report = build_drag_report(ac)
print(report.cd0, report.k)
for term in report.parasite_terms:
    print(term.name, term.value, term.method)
```

The report computes its own reference condition from the aircraft ceiling and
cruise speed (`report.reference_mach`, `report.reference_altitude_ft`). It
does not read the reference condition stored in the record; the tool
`tools/update_aircraft_drag_models.py` writes the report values into the
records. The report also lists the geometry fields that were estimated because
the record has no value for them (`report.imputed_fields`).

## Parabolic polar

For simple analyses, `build_reference_drag_polar` reduces the model to
`CD = cd0 + k CL²` at the reference condition stored in the record
(`drag.reference_mach` and `drag.reference_reynolds_per_m`):

```python
from aerox.models.drag import build_reference_drag_polar

polar = build_reference_drag_polar(ac)
print(polar.cd0, polar.k)
print(polar.drag_coefficient(0.55))
```

The polar leaves out wave drag, and it has no Mach or Reynolds number effects.
At high lift coefficients near the reference Mach number it is therefore lower
than the full model: for the A320 at `CL = 1`, 0.0621 against 0.0678. Use
`build_drag_model` when these effects matter.

## Comparison models

`aerox.models.drag.reference` has published drag models for comparison, for
example `PollSchumannDragModel` (Poll and Schumann, 2025). They take the same
arguments as the Aerox model. The Poll-Schumann source uses the aircraft
Reynolds number, based on the square root of the wing area. To evaluate it at
a known aircraft Reynolds number, convert it first:

```
reynolds_per_m = aircraft_reynolds / sqrt(reference_area_m2)
```

## Accuracy

- At cruise, the model agrees with a reference performance model within 5 %
  for most jet transports.
- Near Mach 0.6 it is 4 to 12 % too high. The model has no drag increase with
  Mach below the drag rise yet. See the [roadmap](roadmap.md).
- The record field `drag.confidence` rates the geometry behind each model as
  medium, medium-low, or low.
