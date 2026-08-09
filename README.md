# Aerox

Aerox is a new aircraft performance data library and the successor project to
OpenAP. The first version is intentionally small: it packages curated aircraft
YAML files and exposes typed Python helpers for loading them.

## Current Scope

- `aerox.aircraft` dataclasses for aircraft configuration data
- `aerox.aero` NumPy atmosphere and IAS/CAS/TAS/Mach conversions ported from a
  pinned OpenAP implementation
- `aerox.fms` local spatial waypoint, altitude/speed profile, wind-correction,
  and online look-ahead guidance primitives
- bundled aircraft YAML data under `src/aerox/data/aircraft/yml`
- a local Flask editor for maintaining the aircraft YAML files

Experimental QAR decoding, kinematic replay, identification, translational
force reconstruction, effective propulsion/lift/fuel calibration, a bounded
commanded-N1 actuator, a 10 Hz longitudinal replay model, and a pitch–bank–
thrust 3D point-mass model live under `aerox.flightdata`, `aerox.models`,
`aerox.modeling`, `aerox.guidance`, and `tools/`; their evidence and limitations
are tracked in `docs/modeling`. The point-mass model has completed its first A320
QAR panel replay: short-window longitudinal behavior improves, while bank-only
heading propagation fails the multi-hour open-loop boundary. A subsequent
observed yaw-rate audit strongly improves replay, but inconsistent effective
rate scales limit that branch to diagnostics rather than a physical yaw-control
model. The replay-only adapter is implemented with an explicit implied turn-
force residual; bank remains the trajectory-generation control. Lateral-
acceleration and beta-proxy attribution do not pass the physical release gate,
so A320 lateral dynamics now require direct sideslip data or an external prior.
The first public-source prior gate added an EASA-certified A320-214 identity,
mass limits, and corrected span/MAC to the compact aircraft record. External
geometry and higher-order priors remain modeling artifacts rather than hundreds
of mostly empty runtime YAML fields; generic NASA transport coefficients remain
reference values rather than A320 parameters.
The subsequent OpenVSP/VSPAERO gate now provides a mesh-converged, CAD-planform
static derivative prior and a directly inspectable A320 development model. It
remains inviscid and omits proprietary airfoils, twist, fuselage/nacelle
interference, aeroelasticity, and control laws, so it does not replace the
viscous drag polar or constitute a released physical six-degree-of-freedom
aircraft model.
The same documentation also includes the RJ85/DASHlink archive/cadence audit and
its first kinematic AOA-decoding experiment.

The first deterministic FMS milestone converts local waypoint intent with
altitude and speed constraints into online references for the A320
pitch–bank–realized-net-thrust plant. Its architecture, comparison trajectory,
gates, and limitations are recorded in `docs/fms`. A leakage-safe multi-flight
gate also releases a small phase-conditioned CAS/Mach target surrogate for the
TOC/TOD trajectory layer. Its first unchanged-plant integration now includes a
generated fuel/mass planning iteration; target inference transfers, but the
closed-loop profile remains held on speed/energy timing and climb feasibility.

The only OpenAP-derived fuel behavior currently included is a pinned,
dependency-light diagnostic curve. General emissions, BADA, CasADi
optimization, geodetic FMS legs, and four-dimensional arrival-time planning
are not yet part of the runtime package.

## Usage

```python
from aerox import load_aircraft

aircraft = load_aircraft("AT72")
print(aircraft.mtow)
```

## Data Editor

From the repository root:

```bash
uv run --extra editor tools/web_editor.py
```

Then open `http://localhost:5000`.

Research scripts that use pandas, PyArrow, or Matplotlib can be run with the
`research` extra. A normal Aerox installation includes only runtime modeling
dependencies.

## Tests

```bash
uv run pytest -q
```
