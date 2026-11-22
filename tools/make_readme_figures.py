"""Make the figures used in the README and the docs.

    uv run --extra research tools/make_readme_figures.py

The figures are written to ``docs/figures``. They use only the public API.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402

from aerox import aero, get_available_aircraft, load_aircraft  # noqa: E402
from aerox.models.drag import build_drag_model, build_parabolic_polar  # noqa: E402
from aerox.models.drag.builder import resolve_drag_geometry  # noqa: E402

OUT = Path(__file__).parents[1] / "docs" / "figures"

# Categorical slots, in fixed order.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6",
    "#eb6834",
    "#1baf7a",
    "#eda100",
    "#e87ba4",
    "#008300",
    "#6250d6",
    "#e34948",
)
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
    }
)


def cruise_cl(ac, mass_fraction=0.85):
    """Lift coefficient at the reference condition and a typical mass."""

    h = ac.drag.reference_altitude_ft * aero.ft
    v = aero.mach2tas(ac.drag.reference_mach, h)
    q = 0.5 * aero.density(h) * v**2
    return mass_fraction * ac.mtow * aero.g0 / (q * ac.wing.area)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(OUT / name)


def fig_polar():
    ac = load_aircraft("A320")
    model = build_drag_model(ac)
    polar = build_parabolic_polar(ac)
    cl = np.linspace(0.0, 1.0, 101)
    h = 11_000

    fig, ax = plt.subplots(figsize=(7, 4.2))
    for mach, color in [(0.5, BLUE), (0.78, ORANGE), (0.82, AQUA)]:
        ax.plot(
            model.cd(cl=cl, mach=mach, h=h),
            cl,
            color=color,
            label=f"full model, Mach {mach}",
        )
    ax.plot(
        polar.cd(cl=cl),
        cl,
        color=INK2,
        linestyle="--",
        linewidth=1.5,
        label="parabolic polar, cd0 + k CL²",
    )
    ax.set_xlabel("Drag coefficient CD")
    ax.set_ylabel("Lift coefficient CL")
    ax.set_title("A320 drag polar at 11 000 m")
    ax.legend(loc="lower right")
    save(fig, "drag-polar.png")


def fig_drag_rise():
    fig, ax = plt.subplots(figsize=(7, 4.2))
    cl = 0.5
    aircraft = [("A320", BLUE), ("B738", ORANGE), ("B789", AQUA), ("A388", YELLOW)]
    for code, color in aircraft:
        ac = load_aircraft(code)
        model = build_drag_model(ac)
        mach = np.linspace(0.4, ac.mmo + 0.04, 200)
        b = model.breakdown(cl=cl, mach=mach, h=11_000)
        ax.plot(mach, b.total * 1e4, color=color, label=f"{code}, MMO {ac.mmo:.2f}")
        mcrit = float(np.atleast_1d(b.critical_mach)[0])
        y = float(model.cd(cl=cl, mach=mcrit, h=11_000)) * 1e4
        ax.plot(mcrit, y, "o", color=color, markersize=7, mec=SURFACE, mew=2)
    ax.plot([], [], "o", color=INK2, markersize=7, label="critical Mach number")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("CD (drag counts)")
    ax.set_title("Drag rise at CL = 0.5, 11 000 m, up to MMO + 0.04")
    ax.legend(loc="upper left")
    save(fig, "drag-rise.png")


def fig_breakdown():
    codes = ["A320", "B77W", "E190", "AT72", "C172"]
    groups = [
        ("Wing", BLUE),
        ("Fuselage", ORANGE),
        ("Tails", AQUA),
        ("Nacelles and pylons", YELLOW),
        ("Fixed increments", MAGENTA),
        ("Lift-dependent", VIOLET),
        ("Wave", RED),
    ]
    values = {name: [] for name, _ in groups}
    labels = []
    for code in codes:
        ac = load_aircraft(code)
        b = build_drag_model(ac).breakdown(
            cl=cruise_cl(ac),
            mach=ac.drag.reference_mach,
            h=ac.drag.reference_altitude_ft * aero.ft,
        )
        part = {k: float(v) for k, v in b.component_parasite.items()}
        values["Wing"].append(part.get("wing", 0.0))
        values["Fuselage"].append(part.get("fuselage", 0.0))
        values["Tails"].append(
            part.get("horizontal_tail", 0.0) + part.get("vertical_tail", 0.0)
        )
        values["Nacelles and pylons"].append(
            part.get("nacelles", 0.0) + part.get("pylons", 0.0)
        )
        values["Fixed increments"].append(
            sum(float(v) for v in b.parasite_increments.values())
        )
        values["Lift-dependent"].append(
            float(b.span_induced + b.lift_dependent_profile)
        )
        values["Wave"].append(float(getattr(b, "wave", 0.0)))
        labels.append(f"{code}\nM {ac.drag.reference_mach:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(codes))
    bottom = np.zeros(len(codes))
    for name, color in groups:
        v = np.array(values[name]) * 1e4
        ax.bar(
            x,
            v,
            0.6,
            bottom=bottom,
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=1.5,
        )
        bottom += v
    for xi, total in zip(x, bottom, strict=True):
        ax.annotate(
            f"{total:.0f}",
            (xi, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
        )
    ax.set_xticks(x, labels)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("CD (drag counts)")
    ax.set_title("Drag breakdown at the reference cruise condition")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(fig, "drag-breakdown.png")


def fig_fleet():
    fig, ax = plt.subplots(figsize=(7, 4.4))
    rows = []
    for code in get_available_aircraft():
        ac = load_aircraft(code)
        if ac.engine.type != "turbofan":
            continue
        polar = build_parabolic_polar(ac)
        geometry = resolve_drag_geometry(ac)
        estimated = "wing.area" in geometry.imputed_fields or not ac.wing.area
        ld_max = 1.0 / (2.0 * np.sqrt(polar.cd0 * polar.k))
        rows.append((geometry.aspect_ratio, ld_max, estimated, code))
    known = [r for r in rows if not r[2]]
    guessed = [r for r in rows if r[2]]
    ax.scatter(
        [r[0] for r in known],
        [r[1] for r in known],
        s=36,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.0,
        label=f"turbofan aircraft ({len(rows)})",
    )
    if guessed:
        ax.scatter(
            [r[0] for r in guessed],
            [r[1] for r in guessed],
            s=36,
            facecolor="none",
            edgecolor=BLUE,
            linewidth=1.5,
            label="wing area estimated",
        )
    for aspect, ld, _, code in rows:
        if code in {"A388", "B789", "A320", "B738", "F16", "CRJ9", "B744"}:
            ax.annotate(
                code,
                (aspect, ld),
                xytext=(5, 2),
                textcoords="offset points",
                fontsize=8,
                color=INK,
            )
    ax.set_xlabel("Wing aspect ratio")
    ax.set_ylabel("Maximum lift-to-drag ratio")
    ax.set_title("Turbofan aircraft: maximum L/D from the parabolic polar")
    ax.legend(loc="lower right")
    save(fig, "fleet-lift-to-drag.png")


def _trapezoid(x_apex, span, area, taper, sweep_c4_deg, mirror=True):
    """Outline of a trapezoidal planform seen from above, x pointing aft."""

    root = 2.0 * area / (span * (1.0 + taper))
    tip = taper * root
    semi = span / 2.0
    tan_le = np.tan(np.deg2rad(sweep_c4_deg)) + (root - tip) / (4.0 * semi)
    x_tip = x_apex + semi * tan_le
    right = [(x_apex, 0.0), (x_tip, semi), (x_tip + tip, semi), (x_apex + root, 0.0)]
    if not mirror:
        return right
    left = [(x, -y) for x, y in reversed(right)]
    return right + left


def _mac_quarter_chord(span, area, taper, sweep_c4_deg):
    root = 2.0 * area / (span * (1.0 + taper))
    y_mac = span / 6.0 * (1.0 + 2.0 * taper) / (1.0 + taper)
    mac = 2.0 / 3.0 * root * (1 + taper + taper**2) / (1 + taper)
    tan_le = np.tan(np.deg2rad(sweep_c4_deg)) + (root - taper * root) / (2.0 * span)
    return y_mac * tan_le + 0.25 * mac


def _planform_shapes(ac):
    """Fuselage, wing and horizontal tail outlines, nose at x = 0, x aft."""

    wing, ht, fus = ac.wing, ac.horizontal_tail, ac.fuselage
    taper = wing.taper_ratio or 0.25
    x_wing_ac = 0.45 * fus.length
    x_apex = x_wing_ac - _mac_quarter_chord(wing.span, wing.area, taper, wing.sweep)

    r = fus.width / 2
    nose, tail = 2.0 * fus.width, 3.0 * fus.width
    t = np.linspace(0, np.pi / 2, 20)
    body = (
        [(nose - nose * np.cos(a), r * np.sin(a)) for a in t]
        + [(fus.length - tail, r), (fus.length, 0.15 * r)]
        + [(fus.length, -0.15 * r), (fus.length - tail, -r)]
        + [(nose - nose * np.cos(a), -r * np.sin(a)) for a in t[::-1]]
    )
    shapes = [
        (body, INK2, 0.3),
        (_trapezoid(x_apex, wing.span, wing.area, taper, wing.sweep), BLUE, 0.9),
    ]
    tail_placed = False
    if ht.area and ht.span:
        ht_taper = ht.taper_ratio or 0.35
        ht_sweep = ht.sweep if ht.sweep is not None else wing.sweep + 5
        root = 2.0 * ht.area / (ht.span * (1.0 + ht_taper))
        if ht.arm:
            x_ht = (
                x_wing_ac
                + ht.arm
                - _mac_quarter_chord(ht.span, ht.area, ht_taper, ht_sweep)
            )
        else:
            x_ht = fus.length - 0.3 * tail - root
            tail_placed = True
        shapes.append(
            (_trapezoid(x_ht, ht.span, ht.area, ht_taper, ht_sweep), AQUA, 0.9)
        )
    return shapes, tail_placed


def fig_planforms():
    codes = ["E190", "A320", "B789", "A388"]
    aircraft = [load_aircraft(c) for c in codes]
    gap = 6.0
    length = max(ac.fuselage.length for ac in aircraft)
    width = sum(ac.wing.span for ac in aircraft) + gap * (len(aircraft) - 1)
    x_range = (-1.0, width + 1.0)
    y_range = (-length - 9.0, 7.0)
    aspect = (y_range[1] - y_range[0]) / (x_range[1] - x_range[0])
    fig, ax = plt.subplots(figsize=(10, 10 * aspect + 0.4))
    x0 = 0.0
    for ac in aircraft:
        centre = x0 + ac.wing.span / 2
        shapes, _ = _planform_shapes(ac)
        for outline, color, alpha in shapes:
            ax.add_patch(
                Polygon(
                    [(centre + y, -x) for x, y in outline],
                    closed=True,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor=SURFACE,
                    linewidth=1.0,
                )
            )
        ax.annotate(
            f"{ac.icao}\n{ac.wing.span:.1f} m span",
            (centre, -ac.fuselage.length - 2),
            ha="center",
            va="top",
            fontsize=9,
            color=INK,
        )
        x0 += ac.wing.span + gap
    # scale bar
    ax.plot([0, 10], [4, 4], color=INK, linewidth=2)
    ax.annotate("10 m", (5, 4.5), ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(
        "Wings and horizontal tails drawn from the records, to scale", loc="left"
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=0.92)
    save(fig, "planforms.png")


if __name__ == "__main__":
    fig_polar()
    fig_drag_rise()
    fig_breakdown()
    fig_fleet()
    fig_planforms()
