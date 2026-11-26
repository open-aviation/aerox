# Drag

The drag model computes the drag coefficient of an aircraft from its geometry,
clean or with flaps, slats and landing gear deployed. No value is fitted to another
performance model.

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

| | Civil turbofans | Turboprops, pistons, military jets |
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

A negative Mach number or Reynolds number raises `ValueError`.

## Flaps, slats and landing gear

Give the flap and slat angles in radians. Gear extension runs from 0
(retracted) to 1 (down). Each input defaults to zero.

```python
flap, slat = ac.flaps.angles_rad("FULL")  # A320-214

condition = DragCondition(
    mach=0.2,
    reynolds_per_m=aero.reynolds_per_m(mach=0.2, h=0),
    flap_angle_rad=flap,
    slat_angle_rad=slat,
    gear_extension_fraction=1.0,
)
cd = model.cd(cl=1.5, condition=condition)
```

`angles_rad` returns `None` for the slat angle when the record does not give
it, for example for the CRJ200. `None` means unknown, not retracted. Leave
`slat_angle_rad` out in that case; the result then has no slat drag.

Here `CL = L / (q S)` is the total aircraft lift coefficient in the selected
configuration. `S` is the clean wing reference area. Use the same reference
area at all settings.

The flap model adapts GASP Volume III, equation III.1.58 and section III.1.4.1.
It uses the published GASP tables reproduced in Aviary.
[GASP aerodynamic method](https://ntrs.nasa.gov/citations/19810010564),
[Aviary flap tables](https://github.com/OpenMDAO/Aviary/blob/03bf83e7937f16c65e6667991949f3a68c97e23e/aviary/subsystems/aerodynamics/gasp_based/flaps_model/meta_model.py),
[Aviary flap type constants](https://github.com/OpenMDAO/Aviary/blob/03bf83e7937f16c65e6667991949f3a68c97e23e/aviary/utils/fortran_to_aviary.py),
[Aviary induced drag tables](https://github.com/OpenMDAO/Aviary/blob/03bf83e7937f16c65e6667991949f3a68c97e23e/aviary/subsystems/aerodynamics/gasp_based/gaspaero.py).

```text
ΔCD_flap = DCDOTE × VDEL1(cf/c) × VDEL2(δf / δopt) × f × cos(Λ)
ΔCL_flap = DCLMTE × VLAM3(AR) × VLAM4(t/c) × VLAM5(cf/c)
                 × VLAM6(δf) × f × cos³(Λ)
CD_span  = k_span × (CL − SIGMA(δf) × ΔCL_flap)² / VDEL6(δf)
```

`f = S_wf/S` is the fraction of the wing served by the flaps. It represents
the GASP span and fuselage factors for drag and the span factor for lift.
`Λ` is the wing quarter-chord sweep. It represents flap hinge sweep in the
profile term. The model uses shape-preserving cubic interpolation for
angle-dependent tables. The clean wing span coefficient `k_span` multiplies
the shifted square. This shift captures the effect of flap lift on the
polar at fixed total lift. The net change in span-induced drag can have
either sign. The model uses the clean profile and wave terms at total `CL`.

| Flap type | `δopt` (°) | `DCDOTE` | `DCLMTE` |
| --- | ---: | ---: | ---: |
| Plain | 60 | 0.12 | 0.90 |
| Split | 60 | 0.23 | 0.80 |
| Single-slotted | 40 | 0.13 | 1.18 |
| Double-slotted | 55 | 0.23 | 1.40 |
| Triple-slotted | 55 | 0.23 | 1.60 |
| Fowler | 30 | 0.10 | 1.67 |
| Double-slotted Fowler | 30 | 0.15 | 2.25 |

The two slotted groups with the same drag constants differ in lift correction.
Mixed single- and double-slotted flaps use equal affected area for each type.
When the record gives no flap type, piston aircraft use plain flaps,
turboprops use single-slotted flaps and jets use double-slotted flaps.
The default chord ratio is 0.25 for plain and split flaps and 0.30 for the
other types. Without a stored `f`, the model integrates a trapezoidal wing
between the flap stations. It uses the fuselage side as the inboard station
if needed and a taper ratio of 0.60 for piston aircraft or 0.33 for the
others when taper is missing. If the outboard station is also unknown,
`f` defaults to 0.45 for piston aircraft and 0.65 for the others.

Slat drag uses Torenbeek equation G-62 and Figure G-24. The wing component's
zero-lift drag `CD0_wing` supplies the clean section profile proxy. The
factor `1 + CL²` follows the quadratic section approximation in equation
G-42 with coefficient 1 and zero offset. The assumed slat chord ratio is
0.17. The slatted wing area ratio `S_ws/S` defaults to 0.65. The sine ratio
ramps the drag from zero to the G-62 estimate at the 20° reference angle
used by Aviary. Torenbeek's estimate applies near the lift range where the
slat position is optimum. Actual drag depends on slot gap and incidence.

These parts are aerox assumptions and do not come from the cited sources:
the default slatted area ratio of 0.65, the default taper ratios of 0.60 and
0.33, the equal area split for mixed flaps, the use of `f` for the GASP span
and fuselage factors, and the sine scaling of slat drag with the slat angle.

```text
ΔCD_slat = CD0_wing × (1 + CL²) × (S_ws/S) × 0.17 × cos(Λ)
                    × sin(δs) / sin(20°)
ΔCD_gear = g × (MTOW × g0 / S) × 5.81e−5 × MTOW^(−0.215)
```

Gear drag uses Torenbeek's weight-based correlation as given by Anderson
(1999). MTOW is in kg. It gives 0.0323 for the A320. Deiler and Fezans
report 0.0291 to 0.0311 for the configurations in their flight-data study.
[Deiler and Fezans, Table 5](https://doi.org/10.2514/6.2025-1250).

The breakdown reports the increments as `b.flap`, `b.slat` and `b.gear`.
The lift correction is included in `b.span_induced`. The high-lift terms
apply in low-speed flight, below stall and outside ground effect. Surface
angles outside 0 to 60° and deployed spoilers raise `ValueError`. The gear
term is independent of flap and slat deployment. The parabolic polar and
the comparison models cover the clean configuration only.

### High-lift validation

Against ANP 2.3, the root-mean-square error in total `CD` falls from 0.0182
to 0.0140 for 43 high-lift points with published deflections. Across 174
points, including cases that use detent labels as angle proxies, it falls
from 0.0252 to 0.0230. ANP gives effective drag-to-lift ratios for noise
procedures. The comparison uses the procedure gear state and the speed
coefficient to estimate lift. When the slat angle is unknown, the score
uses zero slat angle.
Slat angles are known for 25 of the 174 points, so the slat term has no
independent validation.
[ANP 2.3 database](https://www.easa.europa.eu/en/domains/environment/policy-support-and-research/aircraft-noise-and-performance-anp-data),
[Directive (EU) 2015/996, Appendix B](https://www.legislation.gov.uk/eudr/2015/996/annex/appendix/B).

For 17 points digitised from Obert Figure 27.35, the 777-200 error falls
from 0.0173 to 0.0117. The score treats detent labels as angles.
For the six A320 ANP points, the model overestimates `CD` by 0.0132 to
0.0266. Obert's linearised efficiencies describe the local slope of the
total polar near 1.2 to 1.4 times stall speed. They serve as a slope
check. [Obert, Chapter 27](https://doi.org/10.3233/978-1-58603-970-7-i).

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

`model.expression()` returns the full model as a closed-form formula that runs
with NumPy, JAX, or CasADi:

```python
import jax.numpy as jnp

expr = build_drag_model(ac).expression()
expr.cd(cl=0.55, mach=0.78, h=11_000, xp=jnp)
```

See [Math backends](backends.md).

## Comparison models

`aerox.models.drag.reference` has published drag models for comparison, for
example `PollSchumannDragModel` (Poll and Schumann, 2025). They take the same
arguments as the aerox model. The Poll-Schumann source uses the aircraft
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
