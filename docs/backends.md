# Math backends: JAX and CasADi

The drag model can be evaluated with NumPy, JAX, CasADi, or another math
library. Use JAX or CasADi for optimization and automatic differentiation, for
example in trajectory optimization.

## The drag expression

`model.expression()` returns the full drag model as a closed-form formula. All
the aircraft-specific values are constants computed when the model is built.
Pass the math library as `xp`: NumPy (the default), `jax.numpy`, or the
`casadi` module.

```python
import casadi
import jax.numpy as jnp
from aerox import load_aircraft
from aerox.models.drag import build_drag_model

expr = build_drag_model(load_aircraft("A320")).expression()

expr.cd(cl=0.55, mach=0.78, h=11_000)                   # NumPy
expr.cd(cl=0.55, mach=0.78, h=11_000, dT=10, xp=jnp)    # JAX, ISA + 10 K
expr.cd(cl=0.55, mach=0.78, h=11_000, xp=casadi)        # CasADi
```

The result equals `model.cd()` with the same inputs. It includes wave drag for
the aircraft that have it.

## Inputs

| Argument | Unit | Notes |
| --- | --- | --- |
| `cl` | | lift coefficient |
| `mach` | | Mach number |
| `h` | m | altitude; give `h` or `reynolds_per_m` |
| `dT` | K | ISA temperature offset, default 0; used with `h` only |
| `reynolds_per_m` | 1/m | Reynolds number per metre; give `h` or `reynolds_per_m` |
| `flap_angle_rad` | rad | flap angle, default clean |
| `slat_angle_rad` | rad | slat angle, default clean |
| `gear_extension_fraction` | | 0 retracted, 1 extended, default retracted |
| `xp` | | math library, default NumPy |

With `h`, the ISA atmosphere and the Reynolds number are computed with `xp`
too, so every input can be a JAX array or a CasADi symbol. To compute the
Reynolds number on its own:

```python
from aerox.models.drag.expression import isa_reynolds_per_m

re = isa_reynolds_per_m(mach=0.78, h=11_000, dT=0, xp=jnp)
expr.cd(cl=0.55, mach=0.78, reynolds_per_m=re, xp=jnp)
```

Flaps, slats and gear in a landing configuration:

```python
expr.cd(
    cl=1.5, mach=0.2, h=0, flap_angle_rad=0.52, slat_angle_rad=0.47,
    gear_extension_fraction=1,
)
```

See [Drag](drag.md#flaps-slats-and-landing-gear) for the configuration model.

## JAX

The expression works with `jax.jit`, `jax.grad`, and `jax.vmap`:

```python
import jax

@jax.jit
def cd(cl, mach, h):
    return expr.cd(cl, mach, h=h, xp=jnp)

cd(0.55, 0.78, 11_000.0)
jax.grad(cd)(0.55, 0.78, 11_000.0)      # dCD/dCL
```

JAX uses 32-bit floats by default. For results that equal NumPy to many
digits, enable 64-bit floats with `jax.config.update("jax_enable_x64", True)`.

## CasADi

Pass the `casadi` module. The inputs can be symbols:

```python
cl, mach, h = casadi.SX.sym("cl"), casadi.SX.sym("mach"), casadi.SX.sym("h")
drag = casadi.Function("cd", [cl, mach, h], [expr.cd(cl, mach, h=h, xp=casadi)])
drag(0.55, 0.78, 11_000)
```

Wave drag and the atmosphere above 11 000 m use `fmax`, which has a kink. For
gradient-based solvers that need smooth functions, pass an object whose
`maximum` is a smooth approximation; OpenAP does this.

## Other math libraries

Another library works if it has `log`, `sqrt`, `exp`, `sin`, and `maximum` (or
`fmax`). `xp` can also be an object with these functions as attributes. A flap
angle also needs `where` (or `if_else`), because the flap curves are piecewise
cubics.

## Limits

- The expression does not check its inputs. The lift and Mach limits of the
  model do not apply. At zero speed, the Reynolds number is zero and the skin
  friction drops to zero, so the result is wrong without an error. Keep the
  Reynolds number above a small minimum, or use `model.cd()` when you need
  these checks.
- The flap curves cover 0 to 60°. The expression holds a flap angle outside
  this range at the nearest limit, so the drag is constant there and its
  gradient is zero. Bound the flap angle in an optimisation problem. The flap
  curves have a continuous first derivative only.
- Only the drag model has an expression. The other functions of aerox, such
  as `aerox.aero`, use NumPy only.
