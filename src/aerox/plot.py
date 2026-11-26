"""Draw an aircraft from its record: a three-view drawing and a 3D view.

Experimental. The geometry is a simple outline model: an elliptic fuselage,
flat trapezoidal wing and tails, and cylindrical nacelles. Sizes come from the
record, completed with the same estimates as the drag model. The records do
not give where the wing and tails sit on the fuselage, or whether the wing is
high or low, so these follow simple rules and are approximate.

Needs Matplotlib (``pip install aerox[plot]``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .models.drag.builder import resolve_drag_geometry, resolve_nacelle_geometry
from .models.drag.components import trapezoidal_root_chord

if TYPE_CHECKING:
    from .aircraft import Aircraft

__all__ = [
    "AircraftGeometry",
    "Part",
    "aircraft_geometry",
    "three_view",
    "view_3d",
    "write_viewer",
]

# Types known to have a high wing. Every other type is drawn with a low wing.
HIGH_WING = {
    "AT43", "AT44", "AT45", "AT72", "AT75", "DH8A", "DH8D", "C130", "C30J",
    "C17", "SH36", "C172", "C182", "C208", "C210", "BE99",
}  # fmt: skip

# Propeller aircraft known to have a T-tail. Rear-engine jets always get one.
T_TAIL = {
    "AT43", "AT44", "AT45", "AT72", "AT75", "DH8A", "DH8D", "E120", "BE20",
    "B350", "B190", "PC12", "PA44", "C17",
}  # fmt: skip

# Military jets with the engines inside the fuselage, drawn with a pointed
# nose, a mid or shoulder wing, and stabilators at the level of the fuselage.
#   wing_z: wing height as a fraction of the fuselage height above its axis
#   fins: number of vertical tails; fin_y: their spacing as a fraction of the
#   fuselage width; strake: a leading-edge strake runs forward to this
#   fraction of the length; tail_gap: the width between the stabilator roots,
#   when known (the fuselage is drawn at least this wide).
FIGHTER = {
    # NASA TM-72861: stabilator span 8.61 m tip to tip, exposed panels 2.39 m.
    "F15": {
        "wing_z": 0.3,
        "fins": 2,
        "fin_y": 0.85,
        "strake": None,
        "tail_gap": 3.82,
    },
    "F16": {"wing_z": 0.05, "fins": 1, "fin_y": 0.0, "strake": 0.27},
    "T38": {"wing_z": -0.15, "fins": 1, "fin_y": 0.0, "strake": None},
    "HAWK": {"wing_z": -0.3, "fins": 1, "fin_y": 0.0, "strake": None},
}

COLORS = {
    "fuselage": "#b9b8b3",
    "wing": "#2a78d6",
    "horizontal_tail": "#1baf7a",
    "vertical_tail": "#1baf7a",
    "nacelle": "#eb6834",
    "propeller": (0.45, 0.45, 0.43, 0.35),
}


@dataclass(frozen=True)
class Part:
    """One part of the aircraft as a list of flat faces, each an (n, 3) array.

    Coordinates are in metres: x aft from the nose, y to the right wing, z up.
    """

    name: str
    faces: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class AircraftGeometry:
    """The parts of an aircraft, and the dimensions used to draw each part.

    ``details`` maps a part name to rows of (label, value, unit, origin), where
    origin is "record", "estimated" (filled as in the drag model), or
    "drawing rule" (an assumption of this drawing).
    """

    icao: str
    parts: tuple[Part, ...]
    details: dict = field(default_factory=dict)

    def points(self) -> np.ndarray:
        return np.vstack([face for part in self.parts for face in part.faces])


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - depends on the install
        raise ImportError(
            "aerox.plot needs Matplotlib: pip install aerox[plot]"
        ) from error
    return plt


# Geometry -------------------------------------------------------------------


def _fighter_fuselage(length, width, height, n_x=48, n_t=28):
    """Elliptic body with a long pointed nose and a straight jet-pipe end."""

    nose = 0.3 * length
    x = np.concatenate(
        [
            nose * (1 - np.cos(np.linspace(0, np.pi / 2, n_x // 2))),
            np.linspace(nose, length, n_x - n_x // 2 + 1)[1:],
        ]
    )
    scale = np.ones_like(x)
    front = x < nose
    # An ogive-like nose: sharp at the tip, full width at the end of the nose.
    scale[front] = np.clip(1 - (1 - x[front] / nose) ** 2, 0, 1) ** 0.75
    rear = x > 0.85 * length
    scale[rear] = 1 - 0.35 * (x[rear] - 0.85 * length) / (0.15 * length)
    t = np.linspace(0, 2 * np.pi, n_t + 1)
    ring = np.stack(
        [
            np.broadcast_to(x[:, None], (len(x), len(t))),
            0.5 * width * scale[:, None] * np.cos(t)[None, :],
            0.5 * height * scale[:, None] * np.sin(t)[None, :],
        ],
        axis=-1,
    )
    faces = [
        np.array([ring[i, j], ring[i + 1, j], ring[i + 1, j + 1], ring[i, j + 1]])
        for i in range(len(x) - 1)
        for j in range(len(t) - 1)
    ]
    faces.append(ring[-1, :-1].copy())  # the nozzle face
    crown = 0.5 * height * scale

    def crown_z(at):
        return float(np.interp(at, x, crown))

    def centre(at):
        return 0.0

    return faces, crown_z, centre


def _fuselage(length, width, height, n_x=48, n_t=28):
    """Elliptic body with a rounded nose and an upswept tail cone."""

    nose, tail = 1.6 * width, min(3.2 * width, 0.4 * length)
    x = np.concatenate(
        [
            nose * (1 - np.cos(np.linspace(0, np.pi / 2, n_x // 3))),
            np.linspace(nose, length - tail, n_x // 3)[1:],
            np.linspace(length - tail, length, n_x - 2 * (n_x // 3) + 1)[1:],
        ]
    )
    scale = np.ones_like(x)
    front = x < nose
    scale[front] = np.sqrt(np.clip(1 - (1 - x[front] / nose) ** 2, 0, 1))
    rear = x > length - tail
    scale[rear] = 1 - 0.88 * (x[rear] - (length - tail)) / tail
    # The tail cone rises so that its top stays close to the crown line.
    centre_z = np.where(rear, (1 - scale) * height * 0.38, 0.0)
    t = np.linspace(0, 2 * np.pi, n_t + 1)
    ring = np.stack(
        [
            np.broadcast_to(x[:, None], (len(x), len(t))),
            0.5 * width * scale[:, None] * np.cos(t)[None, :],
            centre_z[:, None] + 0.5 * height * scale[:, None] * np.sin(t)[None, :],
        ],
        axis=-1,
    )
    faces = [
        np.array([ring[i, j], ring[i + 1, j], ring[i + 1, j + 1], ring[i, j + 1]])
        for i in range(len(x) - 1)
        for j in range(len(t) - 1)
    ]
    crown = centre_z + 0.5 * height * scale

    def crown_z(at):
        return float(np.interp(at, x, crown))

    def centre(at):
        return float(np.interp(at, x, centre_z))

    return faces, crown_z, centre


def _trapezoid(root, tip, half_span, sweep_c4_deg, dihedral_deg=0.0):
    """Leading-edge offset of the tip and the four corners of one half."""

    tan_le = np.tan(np.deg2rad(sweep_c4_deg)) + 0.25 * (root - tip) / half_span
    x_tip = half_span * tan_le
    z_tip = half_span * np.tan(np.deg2rad(dihedral_deg))
    return x_tip, z_tip


def _slab(corners, thickness, axis):
    """Close a flat four-corner outline into a thin solid.

    ``corners`` are LE root, LE tip, TE tip, TE root; ``thickness`` holds the
    root and tip thickness; ``axis`` is the thickness direction (1 or 2).
    """

    offset = np.zeros((4, 3))
    t = np.array([thickness[0], thickness[1], thickness[1], thickness[0]])
    # The thickest point sits near 35 % chord; the edges are thin.
    mid_le = corners[[0, 1]] + 0.35 * (corners[[3, 2]] - corners[[0, 1]])
    offset[:, axis] = 0.5 * t
    up = np.array([corners[0], mid_le[0] + offset[0], corners[3], corners[3]])
    faces = []
    for sign in (1.0, -1.0):
        lift = np.zeros(3)
        root_mid = mid_le[0].copy()
        tip_mid = mid_le[1].copy()
        root_mid[axis] += sign * 0.5 * t[0]
        tip_mid[axis] += sign * 0.5 * t[1]
        faces.append(np.array([corners[0], corners[1], tip_mid, root_mid]))
        faces.append(np.array([root_mid, tip_mid, corners[2], corners[3]]))
    del up, lift
    return faces


def _surface(
    x_apex, z_root, root, tip, half_span, sweep, dihedral, t_c=0.11, y_root=0.0
):
    """Two trapezoidal panels; ``y_root`` moves the panel roots outboard."""

    x_tip, z_tip = _trapezoid(root, tip, half_span, sweep, dihedral)
    right = np.array(
        [
            [x_apex, y_root, z_root],
            [x_apex + x_tip, y_root + half_span, z_root + z_tip],
            [x_apex + x_tip + tip, y_root + half_span, z_root + z_tip],
            [x_apex + root, y_root, z_root],
        ]
    )
    left = right * np.array([1.0, -1.0, 1.0])
    thickness = (t_c * root, t_c * tip)
    return _slab(right, thickness, 2) + _slab(left, thickness, 2)


def _fin(x_apex, z_root, root, tip, height, sweep, t_c=0.10, y=0.0):
    x_tip, _ = _trapezoid(root, tip, height, sweep)
    corners = np.array(
        [
            [x_apex, y, z_root],
            [x_apex + x_tip, y, z_root + height],
            [x_apex + x_tip + tip, y, z_root + height],
            [x_apex + root, y, z_root],
        ]
    )
    return _slab(corners, (t_c * root, t_c * tip), 1)


def _mac_quarter_chord(root, tip, half_span, sweep):
    """Distance from the apex to the quarter-chord point of the MAC, along x."""

    taper = tip / root
    y_mac = half_span / 3 * (1 + 2 * taper) / (1 + taper)
    mac = 2 / 3 * root * (1 + taper + taper**2) / (1 + taper)
    x_tip, _ = _trapezoid(root, tip, half_span, sweep)
    return y_mac / half_span * x_tip + 0.25 * mac


def _cylinder(x_front, y, z, diameter, length, n_t=20, n_x=6):
    t = np.linspace(0, 2 * np.pi, n_t + 1)
    xs = np.linspace(x_front, x_front + length, n_x)
    # Slight taper towards the exhaust.
    radius = 0.5 * diameter * np.linspace(1.0, 0.8, n_x)
    ring = np.stack(
        [
            np.broadcast_to(xs[:, None], (n_x, len(t))),
            y + radius[:, None] * np.cos(t)[None, :],
            z + radius[:, None] * np.sin(t)[None, :],
        ],
        axis=-1,
    )
    sides = [
        np.array([ring[i, j], ring[i + 1, j], ring[i + 1, j + 1], ring[i, j + 1]])
        for i in range(n_x - 1)
        for j in range(n_t)
    ]
    return [ring[0, :-1].copy(), ring[-1, :-1].copy()] + sides


def _disc(x, y, z, diameter, n_t=32):
    t = np.linspace(0, 2 * np.pi, n_t, endpoint=False)
    return np.column_stack(
        [
            np.full(n_t, x),
            y + 0.5 * diameter * np.cos(t),
            z + 0.5 * diameter * np.sin(t),
        ]
    )


def _tail_planform(section, area, mac, default_aspect, default_taper, span_key):
    """Root chord, tip chord and span (or height) of a tail surface.

    When the record has the chords and the area but no span, the chords are
    taken as those of the exposed panels, and the span returned is the
    exposed span (both panels together for a horizontal tail).
    """

    span = getattr(section, span_key, None) if section is not None else None
    root = section.root_chord if section is not None else None
    tip = section.tip_chord if section is not None else None
    if root and tip and span:
        return root, tip, span
    if root and tip and section.area:
        return root, tip, 2 * area / (root + tip)
    if not span:
        span = np.sqrt(default_aspect * area)
        if span_key == "height":
            span = np.sqrt(default_aspect * area)
    root = trapezoidal_root_chord(area, span, mac) if span_key == "span" else None
    if root is None or not np.isfinite(root):
        root = 2 * area / (span * (1 + default_taper))
    tip = max(2 * area / span - root, 0.15 * root)
    return root, tip, span


def aircraft_geometry(aircraft: Aircraft, coarse: bool = False) -> AircraftGeometry:
    """Build the outline geometry of an aircraft from its record.

    ``coarse`` uses fewer faces, for the HTML viewer.
    """

    g = resolve_drag_geometry(aircraft)
    mount = str(aircraft.engine.mount or "").lower() if aircraft.engine else ""
    engines = int(aircraft.engine.number or 0) if aircraft.engine else 0
    rear_engines = "rear" in mount or "aft" in mount or "sides of fuselage" in mount
    high_wing = aircraft.icao in HIGH_WING
    fighter = FIGHTER.get(str(aircraft.icao))
    length, width, height = g.fuselage_length_m, g.fuselage_width_m, g.fuselage_height_m
    widened = bool(fighter and fighter.get("tail_gap"))
    if widened:
        width = max(width, fighter["tail_gap"])

    fuselage_faces, crown_z, centre_z = (_fighter_fuselage if fighter else _fuselage)(
        length, width, height, *((30, 20) if coarse else (48, 28))
    )
    parts = [Part("fuselage", tuple(fuselage_faces))]

    # Wing
    wing = aircraft.wing
    span, area, sweep = g.wing_span_m, g.wing_area_m2, g.wing_sweep_deg
    if wing.root_chord and wing.tip_chord:
        root, tip = wing.root_chord, wing.tip_chord
    else:
        root = trapezoidal_root_chord(area, span, g.wing_mac_m)
        tip = max(2 * area / span - root, 0.1 * root)
    half = span / 2
    if fighter:
        # The wing root ends a little ahead of the stabilators.
        x_apex = 0.75 * length - root
        x_wing_ac = x_apex + _mac_quarter_chord(root, tip, half, sweep)
    else:
        x_wing_ac = (0.5 if rear_engines else 0.45) * length
        x_apex = x_wing_ac - _mac_quarter_chord(root, tip, half, sweep)
    if fighter:
        z_wing = fighter["wing_z"] * 0.5 * height
        dihedral = wing.dihedral if wing.dihedral is not None else 0.0
    elif high_wing:
        z_wing = 0.5 * height + 0.3 * g.wing_thickness_to_chord_ratio * root
        dihedral = wing.dihedral if wing.dihedral else -1.0
    else:
        z_wing, dihedral = -0.32 * height, wing.dihedral if wing.dihedral else 5.0
    t_c = g.wing_thickness_to_chord_ratio
    wing_faces = _surface(x_apex, z_wing, root, tip, half, sweep, dihedral, t_c)
    if fighter and fighter["strake"]:
        # A leading-edge strake from the forward fuselage to the wing root.
        x_tip_le, _ = _trapezoid(root, tip, half, sweep)
        y_join = 0.75 * width
        x_join = x_apex + y_join / half * x_tip_le
        for side in (1.0, -1.0):
            corners = np.array(
                [
                    [fighter["strake"] * length, 0.0, z_wing],
                    [x_join, side * y_join, z_wing],
                    [x_join + 0.3 * root, side * y_join, z_wing],
                    [x_apex + 0.5 * root, 0.0, z_wing],
                ]
            )
            wing_faces += _slab(corners, (0.04 * root, 0.02 * root), 2)
    parts.append(Part("wing", tuple(wing_faces)))

    # Vertical tail
    vt = aircraft.vertical_tail
    fins = fighter["fins"] if fighter else 1
    vt_root, vt_tip, vt_height = _tail_planform(
        vt,
        g.vertical_tail_area_m2 / fins,
        g.vertical_tail_mac_m,
        1.6 if not fighter else 1.2,
        0.35,
        "height",
    )
    vt_sweep = g.vertical_tail_sweep_deg or sweep + 10
    vt_exposed = bool(
        vt is not None and not vt.height and vt.root_chord and vt.tip_chord
    )
    fin_y = [0.0]
    if fighter:
        fin_y = (
            [0.5 * fighter["fin_y"] * width * side for side in (-1, 1)]
            if fins == 2
            else [0.0]
        )
        x_vt = 0.97 * length - vt_root
    elif vt is not None and vt.arm:
        x_vt = (
            x_wing_ac
            + vt.arm
            - _mac_quarter_chord(vt_root, vt_tip, vt_height, vt_sweep)
        )
    else:
        x_vt = 0.985 * length - vt_root - 0.15 * vt_height
    z_vt = crown_z(x_vt + 0.5 * vt_root) - 0.05 * height
    if fins == 2:
        z_vt *= np.sqrt(max(1 - fighter["fin_y"] ** 2, 0.0))
    fin_faces = []
    for y in fin_y:
        fin_faces += _fin(x_vt, z_vt, vt_root, vt_tip, vt_height, vt_sweep, y=y)
    parts.append(Part("vertical_tail", tuple(fin_faces)))

    # Horizontal tail: a T-tail on rear-engine jets, otherwise on the tail cone.
    ht = aircraft.horizontal_tail
    ht_root, ht_tip, ht_span = _tail_planform(
        ht, g.horizontal_tail_area_m2, g.horizontal_tail_mac_m, 4.5, 0.4, "span"
    )
    ht_sweep = g.horizontal_tail_sweep_deg or sweep + 5
    ht_half = ht_span / 2
    ht_exposed = bool(ht is not None and not ht.span and ht.root_chord and ht.tip_chord)
    ht_y_root = 0.0
    if ht_exposed:
        gap = fighter.get("tail_gap") if fighter else None
        ht_y_root = 0.5 * gap if gap else 0.4 * width
    t_tail = not fighter and (
        (rear_engines and aircraft.engine.type == "turbofan") or aircraft.icao in T_TAIL
    )
    if fighter:
        # Stabilators beside the jet pipe, ending at the end of the fuselage.
        x_ht = length - ht_root
        z_ht = 0.0
    elif t_tail:
        fin_x_tip, _ = _trapezoid(vt_root, vt_tip, vt_height, vt_sweep)
        x_ht = x_vt + fin_x_tip + 0.5 * vt_tip - 0.3 * ht_root
        z_ht = z_vt + vt_height
    elif ht is not None and ht.arm:
        x_ht = (
            x_wing_ac + ht.arm - _mac_quarter_chord(ht_root, ht_tip, ht_half, ht_sweep)
        )
        z_ht = centre_z(x_ht + 0.5 * ht_root) + 0.1 * height
    else:
        x_ht = 0.99 * length - ht_root - 0.1 * ht_half
        z_ht = centre_z(x_ht + 0.5 * ht_root) + 0.1 * height
    ht_dihedral = (
        (ht.dihedral if ht is not None and ht.dihedral else 0.0) if not t_tail else -2.0
    )
    parts.append(
        Part(
            "horizontal_tail",
            tuple(
                _surface(
                    x_ht,
                    z_ht,
                    ht_root,
                    ht_tip,
                    ht_half,
                    ht_sweep,
                    ht_dihedral,
                    y_root=ht_y_root,
                )
            ),
        )
    )

    # Nacelles
    nacelle_faces, propeller_faces = [], []
    propeller = aircraft.engine.type in ("turboprop", "piston")
    if propeller and engines == 1 and not mount:
        mount = "nose"  # a single propeller engine without a recorded mount
    if propeller and "nose" in mount:
        propeller_faces.append(_disc(-0.05, 0.0, 0.0, 1.3 * height))
    if engines and not fighter and "in fuselage" not in mount and "nose" not in mount:
        diameter = n_length = None
        if propeller:
            # Nacelle size from the mass per engine; an outline estimate only.
            mass = float(aircraft.mtow or 5000.0) / engines
            diameter = 0.057 * mass ** (1 / 3)
            n_length = 3.0 * diameter
        else:
            try:
                nacelle = resolve_nacelle_geometry(aircraft)
                diameter, n_length = nacelle.max_diameter_m, nacelle.length_m
            except ValueError:
                pass
        if diameter:
            x_tip, z_tip = _trapezoid(root, tip, half, sweep, dihedral)

            def wing_at(eta):
                x_le = x_apex + eta * x_tip
                chord = root + eta * (tip - root)
                return x_le, chord, z_wing + eta * z_tip

            positions = []
            on_wing = engines - (1 if "fin" in mount else 0)
            if rear_engines:
                x_front = 0.82 * length - n_length
                for side in (-1, 1):
                    positions.append(
                        (x_front, side * (0.5 * width + 0.65 * diameter), 0.2 * height)
                    )
            elif on_wing > 0:
                stations = {1: [0.0], 2: [0.34], 4: [0.33, 0.62]}.get(
                    on_wing, list(np.linspace(0.3, 0.6, on_wing // 2))
                )
                for eta in stations:
                    x_le, chord, z_here = wing_at(eta)
                    if aircraft.engine.type == "turbofan":
                        x_front = x_le - 0.55 * n_length
                        z = z_here - 0.65 * diameter
                    else:
                        x_front = x_le - 0.45 * n_length
                        z = z_here + (-0.3 if high_wing else 0.2) * diameter
                    for side in (-1, 1):
                        positions.append((x_front, side * eta * half, z))
            if "fin" in mount:
                positions.append((x_vt - 0.6 * n_length, 0.0, z_vt + 0.55 * diameter))
            for x_front, y, z in positions:
                nacelle_faces += _cylinder(
                    x_front, y, z, diameter, n_length, *((14, 3) if coarse else (20, 6))
                )
                if propeller:
                    propeller_faces.append(_disc(x_front - 0.1, y, z, 3.0 * diameter))
    if nacelle_faces:
        parts.append(Part("nacelle", tuple(nacelle_faces)))
    if propeller_faces:
        parts.append(Part("propeller", tuple(propeller_faces)))

    # Dimensions used for each part, with where each value comes from.
    imputed = set(g.imputed_fields)

    def origin(field_name, value):
        if value is None or field_name in imputed:
            return "estimated"
        return "record"

    def rule(value):
        return "drawing rule"

    def tail_rows(
        section, kind, area, span_value, span_label, root_, tip_, sweep_, x_le
    ):
        get = (
            (lambda key: getattr(section, key, None))
            if section is not None
            else (lambda key: None)
        )
        span_key = "span" if kind == "horizontal_tail" else "height"
        rows = [
            [
                span_label,
                span_value,
                "m",
                "record"
                if span_label.startswith("Exposed")
                else origin(f"{kind}.{span_key}", get(span_key)),
            ],
            ["Area", area, "m²", origin(f"{kind}.area", get("area"))],
            ["Root chord", root_, "m", origin(f"{kind}.root_chord", get("root_chord"))],
            ["Tip chord", tip_, "m", origin(f"{kind}.tip_chord", get("tip_chord"))],
            [
                "Taper ratio",
                tip_ / root_,
                "",
                origin(f"{kind}.taper_ratio", get("taper_ratio")),
            ],
            ["Quarter-chord sweep", sweep_, "°", origin(f"{kind}.sweep", get("sweep"))],
        ]
        mac_q = x_le + _mac_quarter_chord(
            root_, tip_, span_value / (2 if kind == "horizontal_tail" else 1), sweep_
        )
        arm_value = get("arm")
        rows.append(
            [
                "Arm, wing to tail",
                arm_value if arm_value else mac_q - x_wing_ac,
                "m",
                "record" if arm_value else "drawing rule",
            ]
        )
        return rows

    details = {
        "fuselage": {
            "title": "Fuselage",
            "rows": [
                [
                    "Length",
                    length,
                    "m",
                    origin("fuselage.length", aircraft.fuselage.length),
                ],
                [
                    "Width",
                    width,
                    "m",
                    "drawing rule"
                    if widened
                    else origin("fuselage.width", aircraft.fuselage.width),
                ],
                [
                    "Height",
                    height,
                    "m",
                    origin("fuselage.height", aircraft.fuselage.height),
                ],
            ],
        },
        "wing": {
            "title": "Wing",
            "rows": [
                ["Span", span, "m", origin("wing.span", wing.span)],
                ["Area", area, "m²", origin("wing.area", wing.area)],
                ["Aspect ratio", span**2 / area, "", origin("wing.area", wing.area)],
                [
                    "Mean aerodynamic chord",
                    g.wing_mac_m,
                    "m",
                    origin("wing.mac", wing.mac),
                ],
                ["Root chord", root, "m", origin("wing.root_chord", wing.root_chord)],
                ["Tip chord", tip, "m", origin("wing.tip_chord", wing.tip_chord)],
                [
                    "Taper ratio",
                    tip / root,
                    "",
                    origin("wing.taper_ratio", wing.taper_ratio),
                ],
                ["Quarter-chord sweep", sweep, "°", origin("wing.sweep", wing.sweep)],
                ["Thickness ratio t/c", t_c, "", origin("wing.t/c", wing.t_c)],
                [
                    "Dihedral",
                    dihedral,
                    "°",
                    "record" if wing.dihedral else "drawing rule",
                ],
                [
                    "Position",
                    (
                        "mid or shoulder wing"
                        if fighter
                        else "high wing"
                        if high_wing
                        else "low wing"
                    ),
                    "",
                    "drawing rule",
                ],
                ["Quarter-MAC from nose", x_wing_ac, "m", "drawing rule"],
            ],
        },
        "horizontal_tail": {
            "title": "Horizontal tail" + (" (T-tail)" if t_tail else ""),
            "rows": tail_rows(
                ht,
                "horizontal_tail",
                g.horizontal_tail_area_m2,
                ht_span,
                "Exposed span" if ht_exposed else "Span",
                ht_root,
                ht_tip,
                ht_sweep,
                x_ht,
            ),
        },
        "vertical_tail": {
            "title": "Vertical tail" + (" (each of two)" if fins == 2 else ""),
            "rows": tail_rows(
                vt,
                "vertical_tail",
                g.vertical_tail_area_m2 / fins,
                vt_height,
                "Exposed height" if vt_exposed else "Height",
                vt_root,
                vt_tip,
                vt_sweep,
                x_vt,
            ),
        },
    }
    if nacelle_faces:
        if propeller:
            n_origin = "drawing rule"
        else:
            n_origin = "record" if nacelle.tier.startswith("A") else "estimated"
        details["nacelle"] = {
            "title": "Engines and nacelles",
            "rows": [
                ["Engine", str(aircraft.engine.default or "unknown"), "", "record"],
                ["Number of engines", engines, "", "record"],
                ["Mount", str(aircraft.engine.mount or "unknown"), "", "record"],
                ["Nacelle diameter", diameter, "m", n_origin],
                ["Nacelle length", n_length, "m", n_origin],
            ]
            + (
                [["Fan diameter", nacelle.fan_diameter_m, "m", "record"]]
                if not propeller
                else []
            )
            + [
                [
                    "Position",
                    "under the wing" if not rear_engines else "rear fuselage",
                    "",
                    "drawing rule",
                ]
            ],
        }
    if propeller_faces:
        details["propeller"] = {
            "title": "Propellers",
            "rows": [
                ["Number", len(propeller_faces), "", "record"],
                [
                    "Diameter",
                    (3.0 * diameter) if nacelle_faces else 1.3 * height,
                    "m",
                    "drawing rule",
                ],
            ],
        }
    del rule

    return AircraftGeometry(str(aircraft.icao), tuple(parts), details)


# Drawing --------------------------------------------------------------------


def _project(face, view):
    """Project a face to 2D, and return its depth (larger is nearer)."""

    x, y, z = face[:, 0], face[:, 1], face[:, 2]
    if view == "top":  # from above, nose to the left, right wing up
        return np.column_stack([x, y]), z.mean()
    if view == "side":  # from the left, nose to the left
        return np.column_stack([x, z]), -y.mean()
    # from the front: the right wing appears on the left
    return np.column_stack([-y, z]), -x.mean()


def _view_polygons(geometry, view, offset):
    out = []
    for part in geometry.parts:
        for face in part.faces:
            xy, depth = _project(face, view)
            out.append((depth, xy + offset, COLORS[part.name]))
    return out


def three_view(aircraft: Aircraft, ax=None):
    """Draw the top, side, and front views of an aircraft, to one scale.

    The top view is on the left, the front view to its right, and the side
    view below the top view. Returns the figure.
    """

    plt = _require_matplotlib()
    from matplotlib.collections import PolyCollection

    geometry = aircraft_geometry(aircraft)
    pts = geometry.points()
    (x0, y0, z0), (x1, y1, z1) = pts.min(0), pts.max(0)
    gap = 0.08 * max(x1 - x0, y1 - y0)
    # Place the views: top at the origin, side below it, front to the right.
    side_shift = np.array([0.0, y0 - gap - z1])
    front_shift = np.array([x1 + gap + y1, 0.5 * (y0 + y1) - 0.5 * (z0 + z1)])
    polygons = (
        _view_polygons(geometry, "top", np.zeros(2))
        + _view_polygons(geometry, "side", side_shift)
        + _view_polygons(geometry, "front", front_shift)
    )
    polygons.sort(key=lambda item: item[0])
    width = (x1 - x0) + gap + (y1 - y0)
    height = (y1 - y0) + gap + (z1 - z0)
    if ax is None:
        fig = plt.figure(figsize=(10, 10 * height / width + 0.5))
        ax = fig.add_axes((0, 0, 1, 0.93))
    else:
        fig = ax.figure
    ax.add_collection(
        PolyCollection(
            [item[1] for item in polygons],
            facecolors=[item[2] for item in polygons],
            edgecolors=[item[2] for item in polygons],
            linewidths=0.4,
        )
    )
    ax.set_xlim(x0 - 0.02 * width, x0 + 1.02 * width)
    ax.set_ylim(y0 - gap - (z1 - z0) - 0.02 * height, y1 + 0.02 * height)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(f"{aircraft.aircraft or aircraft.icao}", loc="left")
    return fig


def view_3d(aircraft: Aircraft, elev=22, azim=-125, ax=None):
    """Draw a shaded 3D view of an aircraft. Returns the figure."""

    plt = _require_matplotlib()
    from matplotlib.colors import to_rgb
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    geometry = aircraft_geometry(aircraft)
    if ax is None:
        fig = plt.figure(figsize=(9, 4.8))
        ax = fig.add_axes((0, 0, 1, 1), projection="3d")
    else:
        fig = ax.figure
    light = np.array([-0.3, -0.5, 1.0])
    light /= np.linalg.norm(light)
    faces, shades = [], []
    for part in geometry.parts:
        base = np.array(
            to_rgb(
                COLORS[part.name][:3]
                if isinstance(COLORS[part.name], tuple)
                else COLORS[part.name]
            )
        )
        for face in part.faces:
            normal = np.cross(face[1] - face[0], face[2] - face[0])
            norm = np.linalg.norm(normal)
            brightness = 0.55 + 0.45 * abs(normal @ light) / norm if norm else 0.8
            faces.append(face)
            alpha = 0.35 if part.name == "propeller" else 1.0
            shades.append((*np.clip(base * brightness, 0, 1), alpha))
    ax.add_collection3d(
        Poly3DCollection(faces, facecolors=shades, edgecolors=shades, linewidths=0.2)
    )
    pts = geometry.points()
    lo, hi = pts.min(0), pts.max(0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.set_box_aspect(hi - lo, zoom=1.3)
    ax.view_init(elev=elev, azim=azim)
    ax.invert_xaxis()
    ax.set_axis_off()
    ax.set_title(f"{aircraft.aircraft or aircraft.icao}", loc="left")
    return fig


# HTML viewer ----------------------------------------------------------------


def _mesh(part: Part) -> dict:
    """Indexed triangle mesh of a part, in centimetres, with shared vertices."""

    triangles = []
    for face in part.faces:
        for k in range(1, len(face) - 1):
            triangles.append(face[[0, k, k + 1]])
    points = np.round(np.concatenate(triangles) * 100).astype(np.int64)
    unique, index = np.unique(points, axis=0, return_inverse=True)
    # Viewer axes: x aft, y up, z to the left wing (a right-handed rotation).
    vertices = np.column_stack([unique[:, 0], unique[:, 2], -unique[:, 1]])
    return {"v": vertices.ravel().tolist(), "i": index.ravel().tolist()}


def _plain(value):
    """A JSON value: rounded floats, plain ints and strings."""

    if isinstance(value, (str, int)) and not isinstance(value, bool):
        return value
    return round(float(value), 4)


def _summary(aircraft) -> list:
    engine = aircraft.engine
    rows = [
        ["Type", aircraft.aircraft or aircraft.icao, ""],
        ["ICAO code", aircraft.icao, ""],
        ["MTOW", aircraft.mtow, "kg"],
        ["OEW", aircraft.oew, "kg"],
        ["Maximum passengers", aircraft.pax.max if aircraft.pax else None, ""],
        ["MMO", aircraft.mmo, ""],
        ["VMO", aircraft.vmo, "kt"],
        ["Ceiling", aircraft.ceiling, "ft"],
        [
            "Engines",
            f"{engine.number} × {engine.default}" if engine and engine.number else None,
            "",
        ],
    ]
    return [
        [label, _plain(value), unit] for label, value, unit in rows if value is not None
    ]


def _flatten(value, path: str = "", empty: bool = True):
    """Yield ``(path, value)`` for each leaf, with the YAML field names.

    Empty leaves give ``None``, so that the data view can list what is
    missing, except inside named entries such as the flap settings, whose
    optional fields would only add noise.
    """

    from dataclasses import fields, is_dataclass

    if is_dataclass(value):
        value = {
            item.metadata.get("yaml_key", item.name): getattr(value, item.name)
            for item in fields(value)
        }
    elif isinstance(value, dict):
        empty = False
    if isinstance(value, dict) and value:
        for key, child in value.items():
            if child is not None or empty:
                yield from _flatten(child, f"{path}.{key}" if path else str(key), empty)
    elif isinstance(value, (list, tuple)) and value:
        if all(hasattr(item, "subtype") for item in value):
            for item in value:
                yield f"{path}.{item.subtype}", ", ".join(item.engines)
        else:
            yield path, ", ".join(str(item) for item in value)
    elif isinstance(value, (dict, list, tuple)):
        yield path, None
    else:
        yield path, value


# Source-note keys shown in their own columns, not in the row details.
_NOTE_COLUMNS = ("value", "unit", "status", "source")


def _record_rows(aircraft) -> list:
    """Every record field with its source note, for the data view.

    Each row is ``[path, value, unit, status, source, details]``. ``value``
    is ``None`` for an empty field. ``details`` holds the other source-note
    entries, such as ``page_or_section`` and ``notes``. Values without a
    source note have empty strings and an empty dict in the last four places.
    Paths with a note but no field in the record (such as a flap setting the
    record does not have) are added as empty rows.
    """

    import typing
    from dataclasses import fields, is_dataclass

    notes = aircraft.data_provenance.get("sidecar", {}).get("fields", {})
    hints = typing.get_type_hints(type(aircraft))
    # These groups repeat other values or hold the notes themselves.
    skip = {"data_provenance", "certified_mass_limits"}
    leaves = []
    for item in fields(aircraft):
        if item.name in skip:
            continue
        value = getattr(aircraft, item.name)
        if value is None:
            cls = next(
                (arg for arg in typing.get_args(hints[item.name]) if is_dataclass(arg)),
                None,
            )
            value = cls() if cls else None
        # A source id repeats the source of the note.
        leaves.extend(
            (path, value)
            for path, value in _flatten(value, item.name)
            if not path.endswith(".source_id")
        )
    paths = {path for path, _ in leaves}
    for path in notes:
        if path not in paths and not any(p.startswith(f"{path}.") for p in paths):
            leaves.append((path, None))

    rows = []
    for path, value in leaves:
        # A note can cover a whole group, such as the subtype list.
        parts = path.split(".")
        prefixes = (".".join(parts[:n]) for n in range(len(parts), 0, -1))
        note = next((notes[p] for p in prefixes if p in notes), {})
        if value is not None and not isinstance(value, (str, bool)):
            value = _plain(value)
        rows.append(
            [
                path,
                value,
                note.get("unit", ""),
                note.get("status", ""),
                "" if note.get("source", "-") == "-" else note["source"],
                {k: v for k, v in note.items() if k not in _NOTE_COLUMNS and v},
            ]
        )
    return rows


def _record_header(aircraft) -> dict:
    """The review date and certification scope of the source note."""

    header = aircraft.data_provenance.get("sidecar", {}).get("header", {})
    scope = header.get("certification_scope", {})
    return {
        "reviewed": header.get("last_reviewed", ""),
        "scope": [[key, value] for key, value in scope.items() if value != "-"],
    }


def _source_documents(aircraft_list) -> dict:
    """Title, date, and link of each cited source, keyed by source id.

    Uses ``sources/sources.yml`` when it exists (in the repository only), and
    the document entries of the source notes otherwise.
    """

    from pathlib import Path

    import yaml

    ledger = {}
    path = Path(__file__).resolve().parents[2] / "sources" / "sources.yml"
    if path.exists():
        for item in yaml.safe_load(path.read_text(encoding="utf-8"))["sources"]:
            date = next(
                (
                    item[key]
                    for key in ("issue_date", "revision_date", "publication_date")
                    if item.get(key)
                ),
                "",
            )
            issue = item.get("revision") or item.get("issue")
            when = [f"issue {issue}"] if issue and issue != "compact-legacy" else []
            when += [str(date)] if date else []
            ledger[item["id"]] = [
                item.get("title", ""),
                ", ".join(when),
                item.get("url") or "",
            ]

    documents = {}
    for aircraft in aircraft_list:
        notes = aircraft.data_provenance.get("sidecar", {}).get("fields", {})
        for note in notes.values():
            ids = [
                s.strip()
                for key in ("source", "searched_source")
                for s in note.get(key, "").split(" | ")
                if s.strip() not in ("", "-")
            ]
            for source in ids:
                if source in ledger:
                    documents[source] = ledger[source]
                    continue
                doc = documents.setdefault(source, ["", "", ""])
                if note.get("source") == source:
                    doc[0] = doc[0] or note.get("document_title", "")
                    doc[1] = doc[1] or note.get("revision_date", "")
                    doc[2] = doc[2] or note.get("source_url", "")
    return documents


def viewer_data(aircraft_list) -> dict:
    """Meshes and record values of several aircraft, keyed by ICAO code."""

    data = {}
    for aircraft in aircraft_list:
        geometry = aircraft_geometry(aircraft, coarse=True)
        data[str(aircraft.icao)] = {
            "name": aircraft.aircraft or aircraft.icao,
            "span": aircraft.wing.span,
            "length": aircraft.fuselage.length,
            "summary": _summary(aircraft),
            "record": _record_rows(aircraft),
            "header": _record_header(aircraft),
            "details": {
                name: {
                    "title": part["title"],
                    "rows": [
                        [label, _plain(value), unit, src]
                        for label, value, unit, src in part["rows"]
                    ],
                }
                for name, part in geometry.details.items()
            },
            "parts": {part.name: _mesh(part) for part in geometry.parts},
        }
    return data


def write_viewer(path, aircraft=None) -> None:
    """Write a self-contained HTML viewer with 2D, 3D, and data views.

    ``aircraft`` is a list of aircraft; all aircraft when ``None``. The page
    loads three.js from a CDN, so the drawings need an internet connection.
    """

    import json
    from importlib.resources import files
    from pathlib import Path

    if aircraft is None:
        from .aircraft import get_available_aircraft, load_aircraft

        aircraft = [load_aircraft(code) for code in get_available_aircraft()]
    data = viewer_data(aircraft)
    # The note details repeat much text: keep each string once, and give the
    # details as a flat list of key and text indices.
    text, index = [], {}

    def ref(string):
        if string not in index:
            index[string] = len(text)
            text.append(string)
        return index[string]

    for entry in data.values():
        for row in entry["record"]:
            row[5] = [ref(string) for item in row[5].items() for string in item]
    payload = json.dumps(data, separators=(",", ":"))
    extra = {
        "__RECORD_TEXT__": json.dumps(text, separators=(",", ":")),
        "__SOURCES__": json.dumps(_source_documents(aircraft), separators=(",", ":")),
    }
    colors = json.dumps(
        {
            name: (
                "#%02x%02x%02x" % tuple(int(255 * c) for c in color[:3])
                if isinstance(color, tuple)
                else color
            )
            for name, color in COLORS.items()
        }
    )
    template = files("aerox").joinpath("viewer.html").read_text(encoding="utf-8")
    for key, value in extra.items():
        template = template.replace(key, value)
    html = template.replace("__AIRCRAFT_DATA__", payload).replace(
        "__PART_COLORS__", colors
    )
    Path(path).write_text(html, encoding="utf-8")
