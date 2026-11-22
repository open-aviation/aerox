# Atmosphere and speeds

`aerox.aero` gives the ISA atmosphere and the airspeed conversions. The
equations and function names are the same as in `openap.aero`.

All inputs and outputs are in SI units: altitude in m, speed in m/s, pressure
in Pa, density in kg/m³, temperature in K. Every function accepts scalars or
NumPy arrays. The optional argument `dT` is the temperature offset from ISA, in
K. It is limited to the range −25 K to +15 K: a larger offset gives the same
result as the limit.

```python
from aerox import aero

p, rho, T = aero.atmos(10_000)
tas = aero.cas2tas(250 * aero.kts, 3_000)
```

## Atmosphere

| Function | Returns |
| --- | --- |
| `atmos(h, dT=0)` | pressure, density, temperature |
| `pressure(h, dT=0)` | pressure |
| `density(h, dT=0)` | density |
| `temperature(h, dT=0)` | temperature |
| `vsound(h, dT=0)` | speed of sound |
| `h_isa(p, dT=0)` | pressure altitude for a pressure |
| `viscosity(h, dT=0)` | dynamic viscosity, Sutherland's law, in Pa s |
| `reynolds_per_m(mach, h, dT=0)` | Reynolds number per metre |

## Speed conversions

| Function | Converts |
| --- | --- |
| `tas2mach(v_tas, h)`, `mach2tas(mach, h)` | TAS and Mach |
| `tas2eas(v_tas, h)`, `eas2tas(v_eas, h)` | TAS and EAS |
| `tas2cas(v_tas, h)`, `cas2tas(v_cas, h)` | TAS and CAS |
| `mach2cas(mach, h)`, `cas2mach(v_cas, h)` | Mach and CAS |
| `crossover_alt(v_cas, mach)` | altitude where a CAS and a Mach number give the same TAS |

All speed conversions also take `dT`.

## Unit constants

`aero.kts` (knot), `aero.ft` (foot), `aero.fpm` (foot per minute), `aero.nm`
(nautical mile), `aero.lbs` (pound), `aero.inch`, and `aero.sqft` convert to SI
units by multiplication. For example, `250 * aero.kts` is 250 kt in m/s.
`aero.g0` is the standard gravity.
