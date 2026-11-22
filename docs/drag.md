# Drag

The drag model computes the drag coefficient of the clean aircraft from its
geometry. No value is fitted to another performance model.

## Model

The drag coefficient has four parts:

```
CD = CD0(Re, M) + CDi(CL) + CDp(CL) + CDw(CL, M)
```

- **Zero-lift drag, `CD0`.** A component build-up. For the wing, the fuselage,
  the horizontal and vertical tails, the nacelles, and the pylons, the model
  multiplies turbulent flat-plate skin friction by a form factor, an
  interference factor, and the wetted area. Skin friction depends on the
  Reynolds number and the Mach number. Fixed increments are added for the
  fuselage upsweep, the control-surface gaps, and the nacelle base. A
  roughness and excrescence allowance adds 9 % of the computed drag.
- **Induced drag, `CDi`.** `CL² / (π A e)`, with the span efficiency factor `e`
  stored in the aircraft record.
- **Lift-dependent profile drag, `CDp`.** Grows with `CD0` and `CL²`.
- **Wave drag, `CDw`.** The Korn equation with Lock's fourth-power law. It is
  zero below the critical Mach number.

The model is identified as `aerox-geometry-drag-v2` and the wave drag as
`aerox-korn-lock-wave-v1` in the `drag` group of each record.

## Evaluate drag at a flight condition

```python
from aerox import aero, load_aircraft
from aerox.models.drag import build_drag_model

ac = load_aircraft("A320")
model = build_drag_model(ac)

reynolds = aero.reynolds_per_m(0.78, 11_000)   # about 5.9e6 per metre
cd = model.drag_coefficient(0.55, mach=0.78, reynolds_per_m=reynolds)
```

`drag_coefficient` takes the lift coefficient, the Mach number, and the
Reynolds number per metre. Each aircraft part applies its own reference length
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
flaps, landing gear, or spoilers raises `ValueError`.

## Drag breakdown

`breakdown` takes the same arguments and returns each part of the drag:

```python
b = model.breakdown(0.55, mach=0.78, reynolds_per_m=reynolds)
print(b.component_parasite)    # zero-lift drag of each aircraft part
print(b.parasite_increments)   # upsweep, gaps, nacelle base, roughness
print(b.span_induced, b.lift_dependent_profile, b.wave)
print(b.critical_mach, b.total)
```

`build_drag_report` gives the zero-lift terms at the reference condition of the
aircraft, together with the method used for each term:

```python
from aerox.models.drag import build_drag_report

report = build_drag_report(ac)
print(report.cd0, report.k)
for term in report.parasite_terms:
    print(term.name, term.value, term.method)
```

The report also lists the geometry fields that were estimated because the
record has no value for them (`report.imputed_fields`).

## Parabolic polar

For simple analyses, `build_reference_drag_polar` reduces the model to
`CD = cd0 + k CL²` at the reference condition of the aircraft:

```python
from aerox.models.drag import build_reference_drag_polar

polar = build_reference_drag_polar(ac)
print(polar.cd0, polar.k)
print(polar.drag_coefficient(0.55))
```

The polar has no Mach or Reynolds number effects. Use `build_drag_model` when
these matter.

## Accuracy

- At cruise, the model agrees with a reference performance model within 5 %
  for most jet transports.
- Near Mach 0.6 it is 4 to 12 % too high. The model has no drag increase with
  Mach below the drag rise yet. See the [roadmap](roadmap.md).
- The record field `drag.confidence` rates the geometry behind each model as
  medium, medium-low, or low.
