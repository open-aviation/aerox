# Drag

The drag model computes the drag coefficient of the clean aircraft from its
geometry. No value is fitted to another performance model.

## Two ways to get drag

| | `build_drag_model(ac)` | `build_parabolic_polar(ac)` |
| --- | --- | --- |
| Returns | the full model, `GeometryDragModel` or `TransonicGeometryDragModel` | `ParabolicDragPolar`, `CD = cd0 + k CL²` |
| Zero-lift drag | changes with the Mach and Reynolds number | constant |
| Wave drag | yes, for the aircraft that have it (see below) | no |
| Arguments | `cl`, `mach`, and optionally `h` and `dT`, or `reynolds_per_m` | `cl` only |
| Use it for | performance at a flight condition, Mach sweeps, cruise near the drag rise | quick estimates, closed-form analysis such as the best lift-to-drag ratio, optimisation |

The parabolic polar is derived from the full model: `cd0` and `k` are the full
model without wave drag, evaluated at the reference condition of the record.
Both objects have the same methods, `drag_coefficient(cl, ...)`, its shorthand
`cd(cl, ...)`, and `breakdown(cl, ...)`, so code that takes a drag model
accepts either one.

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

| | Civil turbofans (93 types) | Turboprops, pistons, military jets (52 types) |
| --- | --- | --- |
| Parts with skin friction | wing, fuselage, tails, nacelles, pylons | wing, fuselage, tails |
| Fixed increments | fuselage upsweep, control-surface gaps, nacelle base, roughness and excrescence (9 %) | one `unresolved` term, 28 to 35 % |
| Wave drag | yes, except the C-17 and KC-135 | no |

The fixed increments are computed once, at the reference condition of the
aircraft, and do not change with the Reynolds or Mach number.

## Full model: drag at a flight condition

```python
from aerox import aero, load_aircraft
from aerox.models.drag import build_drag_model

ac = load_aircraft("A320")
model = build_drag_model(ac)

cd = model.drag_coefficient(cl=0.55, mach=0.78, h=11_000)
cd = model.cd(cl=0.55, mach=0.78, h=11_000)            # the same, shorter
cd = model.cd(cl=0.55, mach=0.78, h=11_000, dT=10)     # ISA + 10 K
```

`drag_coefficient`, or its shorthand `cd`, takes the lift coefficient `cl` and
the Mach number `mach`. The Reynolds number can be given in one of three ways:

- `h=`, the altitude in m, and optionally `dT=`, the temperature offset from
  ISA in K. The model computes the Reynolds number per metre in the ISA
  atmosphere with `aero.reynolds_per_m(mach, h, dT)`, which uses Sutherland's
  law for the viscosity.
- `reynolds_per_m=`, the Reynolds number per metre, if you have it from
  another source:

  ```python
  reynolds = aero.reynolds_per_m(mach=0.78, h=11_000)   # about 5.9e6 per metre
  cd = model.cd(cl=0.55, mach=0.78, reynolds_per_m=reynolds)
  ```

- Neither: the model uses the reference value stored in the aircraft record
  (`drag.reference_reynolds_per_m`).

Giving both `h` and `reynolds_per_m` raises `TypeError`. Each aircraft part
applies its own reference length to the Reynolds number per metre. All
arguments accept NumPy arrays and broadcast against each other.

The flight condition can also be passed as one object:

```python
from aerox.models.drag import DragCondition

condition = DragCondition(mach=0.78, reynolds_per_m=reynolds)
cd = model.cd(cl=0.55, condition=condition)
```

The model covers the clean configuration only. A condition with deployed
flaps, landing gear, or spoilers raises `ValueError`, and so does a negative
Mach number or Reynolds number.

## Drag breakdown

`breakdown` takes the same arguments and returns each part of the drag:

```python
b = model.breakdown(cl=0.55, mach=0.78, h=11_000)
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

## Parabolic polar: `CD = cd0 + k CL²`

`build_parabolic_polar` reduces the full model to two numbers:

```python
from aerox.models.drag import build_parabolic_polar

polar = build_parabolic_polar(ac)
print(polar.cd0, polar.k)   # A320: 0.0189, 0.0432
print(polar.cd(cl=0.55))    # 0.0320
```

`cd0` and `k` come from the full model without wave drag, at the reference
condition stored in the record (`drag.reference_mach` and
`drag.reference_reynolds_per_m`; for the A320, Mach 0.79).
The polar does not take a Mach or Reynolds number: `polar.cd(cl=0.55,
mach=0.6)` gives the same value as `polar.cd(cl=0.55)`.

How the two compare for the A320:

| Condition | `CL` | Full model | Parabolic polar |
| --- | --- | --- | --- |
| Mach 0.78, 11 000 m | 0.55 | 0.0321 | 0.0320 |
| Mach 0.78, 11 000 m | 1.0 | 0.0660 | 0.0621 |
| Mach 0.6, 6 000 m | 0.55 | 0.0314 | 0.0320 |
| Mach 0.3, 500 m | 0.55 | 0.0323 | 0.0320 |

For jet transports near their cruise condition, the two agree within a few
percent at cruise lift coefficients. The polar is too low at high lift
coefficients near the cruise Mach number, because it has no wave drag, and it
can be off by more than 10 % for small aircraft flown far from their reference
condition, because it keeps the reference Reynolds number. Use
`build_drag_model` when these effects matter.

## Full model in CasADi, JAX, or other math libraries

`model.expression()` returns the full model as a closed-form formula of the
lift coefficient, the Mach number, and the Reynolds number per metre. All the
aircraft-specific values are constants computed when the model is built.
Evaluate it with any math library that has `log`, `sqrt`, and `maximum`:

```python
import jax.numpy as jnp

expr = build_drag_model(ac).expression()
re = aero.reynolds_per_m(mach=0.78, h=11_000)

expr.cd(cl=0.55, mach=0.78, reynolds_per_m=re)              # NumPy
expr.cd(cl=0.55, mach=0.78, reynolds_per_m=re, xp=jnp)      # JAX
```

For CasADi, pass an object whose `maximum` is `casadi.fmax`. The result equals
`model.cd()`, but the expression does not check its inputs, and the lift and
Mach limits of the model do not apply. Use it for optimization and automatic
differentiation, and `model.cd()` otherwise.

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
