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
INK, INK2, MUTED, GRID, AXIS, SURFACE = (
    "#0b0b0b",
    "#52514e",
    "#8a8984",
    "#ebeae6",
    "#c9c8c2",
    "#fcfcfb",
)

plt.rcParams.update(
    {
        "font.family": ["IBM Plex Sans", "DejaVu Sans"],
        "font.size": 9.5,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.labelcolor": INK2,
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    }
)


def titles(ax, title, subtitle):
    """A bold title and a secondary subtitle, left-aligned above the plot."""

    ax.text(
        0,
        1.12,
        title,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="semibold",
        color=INK,
        va="bottom",
    )
    ax.text(
        0, 1.04, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK2, va="bottom"
    )


def end_label(ax, x, y, text, dx=6, dy=0):
    ax.annotate(
        text,
        (x, y),
        xytext=(dx, dy),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=INK,
    )


def cruise_cl(ac, mass_fraction=0.85):
    """Lift coefficient at the reference condition and a typical mass."""

    h = ac.drag.reference_altitude_ft * aero.ft
    v = aero.mach2tas(ac.drag.reference_mach, h)
    q = 0.5 * aero.density(h) * v**2
    return mass_fraction * ac.mtow * aero.g0 / (q * ac.wing.area)


def save(fig, name, tight=True):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        OUT / name, dpi=150, bbox_inches="tight" if tight else None, pad_inches=0.15
    )
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
        cd = model.cd(cl=cl, mach=mach, h=h) * 1e4
        ax.plot(cd, cl, color=color, label=f"Full model, Mach {mach:.2f}")
    cd = polar.cd(cl=cl) * 1e4
    ax.plot(
        cd,
        cl,
        color=INK2,
        linestyle=(0, (4, 3)),
        linewidth=1.5,
        label="Parabolic polar, CD0 + k CL²",
    )
    ax.set_xlabel("Drag coefficient CD (drag counts)")
    ax.set_ylabel("Lift coefficient CL")
    ax.set_ylim(0, 1.05)
    titles(
        ax, "A320 drag polar", "Full drag model at 11 000 m against the parabolic polar"
    )
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
        total = b.total * 1e4
        ax.plot(mach, total, color=color, label=f"{code}, MMO {ac.mmo:.2f}")
        end_label(ax, mach[-1], total[-1], code)
        mcrit = float(np.atleast_1d(b.critical_mach)[0])
        y = float(model.cd(cl=cl, mach=mcrit, h=11_000)) * 1e4
        ax.plot(mcrit, y, "o", color=color, markersize=7, mec=SURFACE, mew=1.5)
    ax.plot([], [], "o", color=MUTED, markersize=7, mec=SURFACE, label="Critical Mach")
    ax.set_xlabel("Mach number")
    ax.set_ylabel("CD (drag counts)")
    ax.margins(x=0.08)
    titles(ax, "Drag rise", "CL = 0.5 at 11 000 m, up to MMO + 0.04")
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
        labels.append(f"{code}\nMach {ac.drag.reference_mach:.2f}")

    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(codes))
    bottom = np.zeros(len(codes))
    handles = []
    for name, color in groups:
        v = np.array(values[name]) * 1e4
        bars = ax.bar(
            x,
            v,
            0.42,
            bottom=bottom,
            color=color,
            label=name,
            edgecolor=SURFACE,
            linewidth=1.0,
        )
        handles.append(bars)
        bottom += v
    for xi, total in zip(x, bottom, strict=True):
        ax.annotate(
            f"{total:.0f}",
            (xi, total),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            color=INK,
            fontsize=9,
            fontweight="semibold",
        )
    ax.set_xticks(x, labels)
    ax.set_ylabel("CD (drag counts)")
    ax.set_ylim(0, bottom.max() * 1.1)
    titles(
        ax,
        "Drag breakdown",
        "By part, at each aircraft's reference cruise condition",
    )
    # Legend in stack order: the top segment first.
    ax.legend(
        handles[::-1],
        [name for name, _ in groups][::-1],
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
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
        s=34,
        color=BLUE,
        edgecolor=SURFACE,
        linewidth=1.2,
        label=f"Wing area from the record ({len(known)})",
        zorder=3,
    )
    if guessed:
        ax.scatter(
            [r[0] for r in guessed],
            [r[1] for r in guessed],
            s=30,
            facecolor=SURFACE,
            edgecolor=BLUE,
            linewidth=1.4,
            label=f"Wing area estimated ({len(guessed)})",
            zorder=3,
        )
    left_labels = {"CRJ9"}
    for aspect, ld, _, code in rows:
        if code in {"A388", "B789", "A320", "B738", "F16", "CRJ9", "B744"}:
            ax.annotate(
                code,
                (aspect, ld),
                xytext=(-6, 3) if code in left_labels else (6, 3),
                ha="right" if code in left_labels else "left",
                textcoords="offset points",
                fontsize=8.5,
                color=INK,
            )
    ax.grid(axis="x")
    ax.set_xlabel("Wing aspect ratio")
    ax.set_ylabel("Maximum lift-to-drag ratio")
    titles(
        ax,
        "Maximum L/D across the turbofan fleet",
        f"{len(rows)} aircraft, from each parabolic polar"
        + ("; hollow markers have an estimated wing area" if guessed else ""),
    )
    save(fig, "fleet-lift-to-drag.png")


def fig_phases(code="A320"):
    """The kinematic data of one aircraft as callouts on a flight profile."""

    ac = load_aircraft(code)
    k = ac.kinematic

    def kt(v):
        return f"{v:,.0f} kt"

    def fpm(v):
        return f"{v:,.0f} ft/min"

    # The profile, in schematic distance and 1 000 ft. The cruise level is
    # schematic; the records do not give one.
    top = 35
    climb_x, climb_h = [0, 1.2, 2.0, 3.1, 4.1, 5.2], [0, 0, 5, 15, 24, top]
    desc_x, desc_h = [8.4, 9.5, 10.6, 11.8, 12.8], [top, 24, 10, 0, 0]
    x = np.array(climb_x + desc_x)
    h = np.array(climb_h + desc_h)

    def climb_at(alt):
        return float(np.interp(alt, climb_h[1:], climb_x[1:]))

    def desc_at(alt):
        return float(np.interp(-alt, [-a for a in desc_h[:-1]], desc_x[:-1]))

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.fill_between(x, h, color=BLUE, alpha=0.08, linewidth=0)
    ax.plot(x, h, color=BLUE, linewidth=2.2, zorder=3)
    ax.plot([-0.4, 13.2], [0, 0], color=AXIS, linewidth=0.8, zorder=1)
    corners = [(2.0, 5), (3.1, 15), (4.1, 24), (5.2, top), (8.4, top), (9.5, 24)]
    corners += [(10.6, 10)]
    ax.plot(
        *zip(*corners, strict=True),
        "o",
        color=BLUE,
        markersize=6.5,
        mec=SURFACE,
        mew=1.5,
        zorder=4,
    )
    for (cx, alt), label in zip(
        corners, ["5 000 ft", "FL150", "FL240", "", "", "FL240", "FL100"], strict=True
    ):
        if label:
            ax.text(cx + 0.18, alt - 0.4, label, fontsize=8, color=MUTED, va="top")

    def callout(name, rows, anchor, at, side):
        """A phase name and its values at ``at``, with a leader to ``anchor``."""

        tx, ty = at
        ha = "right" if side == "left" else "left"
        ax.text(
            tx, ty, name, ha=ha, va="bottom", fontsize=9.5, fontweight="semibold",
            color=INK,
        )  # fmt: skip
        text = "\n".join(f"{label} {value}" for label, value in rows)
        ax.text(
            tx, ty - 0.5, text, ha=ha, va="top", fontsize=9, color=INK2,
            linespacing=1.45,
        )  # fmt: skip
        edge = tx + (0.25 if side == "left" else -0.25)
        ax.plot(
            [edge, anchor[0]],
            [ty + 0.6, anchor[1]],
            color=MUTED,
            linewidth=0.7,
            zorder=2,
        )
        ax.plot(*anchor, "o", color=MUTED, markersize=2.5, zorder=5)

    def mid(x0, h0, x1, h1):
        return (0.5 * (x0 + x1), 0.5 * (h0 + h1))

    left = [
        ("Takeoff", [("V2", kt(k.takeoff_v2)),
                     ("Distance", f"{k.takeoff_distance_m:,} m")],
         (0.6, 0), 3.0),
        ("Initial climb, to 5 000 ft",
         [("IAS", kt(k.climb_to_fl50_ias)), ("ROC", fpm(k.climb_to_fl50_vs))],
         mid(1.2, 0, 2.0, 5), 11.5),
        ("Climb, to FL150",
         [("IAS", kt(k.climb_to_fl150_ias)), ("ROC", fpm(k.climb_to_fl150_vs))],
         mid(2.0, 5, 3.1, 15), 20.0),
        ("Climb, to FL240",
         [("IAS", kt(k.climb_to_fl240_ias)), ("ROC", fpm(k.climb_to_fl240_vs))],
         mid(3.1, 15, 4.1, 24), 28.5),
        ("Mach climb",
         [("Mach", f"{k.mach_climb_mach:.2f}"), ("ROC", fpm(k.mach_climb_vs))],
         mid(4.1, 24, 5.2, top), 37.0),
    ]  # fmt: skip
    for name, rows, anchor, ty in left:
        callout(name, rows, anchor, (min(climb_at(ty), 1.2) - 1.3, ty), "left")

    right = [
        ("Initial descent, to FL240",
         [("Mach", f"{k.descent_to_fl240_mach:.2f}"),
          ("ROD", fpm(k.descent_to_fl240_vs))],
         mid(8.4, top, 9.5, 24), 37.0),
        ("Descent, to FL100",
         [("IAS", kt(k.descent_to_fl100_ias)), ("ROD", fpm(k.descent_to_fl100_vs))],
         mid(9.5, 24, 10.6, 10), 26.0),
        ("Approach",
         [("IAS", kt(k.approach_ias)), ("ROD", fpm(k.approach_vs))],
         mid(10.6, 10, 11.8, 0), 14.5),
        ("Landing",
         [("Speed", kt(k.landing_ias)),
          ("Distance", f"{k.landing_distance_m:,} m")],
         (12.3, 0), 5.0),
    ]  # fmt: skip
    for name, rows, anchor, ty in right:
        callout(name, rows, anchor, (max(desc_at(ty), 11.8) + 1.3, ty), "right")

    # Cruise: above the level segment.
    ax.text(
        6.8, top + 6.2, "Cruise", ha="center", va="bottom", fontsize=9.5,
        fontweight="semibold", color=INK,
    )  # fmt: skip
    ax.text(
        6.8,
        top + 5.7,
        f"Mach {k.cruise_mach:.2f}  ·  TAS {kt(k.cruise_tas)}  ·  "
        f"range {k.cruise_range_nm:,} nm",
        ha="center",
        va="top",
        fontsize=9,
        color=INK2,
    )
    ax.plot([6.8, 6.8], [top + 3.2, top], color=MUTED, linewidth=0.7)
    ax.plot(6.8, top, "o", color=MUTED, markersize=2.5, zorder=5)

    ax.set_xlim(-6.2, 18.2)
    ax.set_ylim(-1.5, top + 12)
    ax.axis("off")
    ax.text(
        -6.2,
        top + 17.5,
        f"{ac.aircraft}: speeds and vertical rates by flight phase",
        fontsize=12,
        fontweight="semibold",
        color=INK,
    )
    ax.text(
        -6.2,
        top + 13.8,
        "From the kinematic data of the record. The profile is schematic. "
        "ROC rate of climb, ROD rate of descent.",
        fontsize=9.5,
        color=INK2,
    )
    save(fig, "kinematics.png")


GALLERY = [
    "A388", "B744", "B789", "A359",
    "B77W", "A333", "B763", "MD11",
    "A320", "B738", "A21N", "B38M",
    "BCS3", "E190", "E75L", "CRJ9",
    "GLF5", "AT72", "DH8D", "C172",
]  # fmt: skip


SHORT_NAMES = {
    "B789": "Boeing 787-9",
    "DH8D": "Dash 8-400",
    "C172": "Cessna 172",
    "MD11": "McDonnell Douglas MD-11",
    "A21N": "Airbus A321neo",
    "E75L": "Embraer E175",
}


def fig_gallery():
    from aerox.plot import view_3d

    cols = 4
    rows = len(GALLERY) // cols
    fig = plt.figure(figsize=(12, 1.8 * rows))
    cell_w, cell_h = 1 / cols, 1 / rows
    for i, code in enumerate(GALLERY):
        ac = load_aircraft(code)
        r, c = divmod(i, cols)
        left, bottom = c * cell_w, 1 - (r + 1) * cell_h
        # The 3D axes is larger than its cell so the aircraft is not cropped;
        # its background is off, so the overlap does not show.
        grow = 0.35
        ax = fig.add_axes(
            (
                left - grow * cell_w / 2,
                bottom - grow * cell_h / 2 + 0.02 * cell_h,
                cell_w * (1 + grow),
                cell_h * (1 + grow),
            ),
            projection="3d",
        )
        view_3d(ac, ax=ax, elev=20, azim=-128)
        extent = [np.ptp(lim) for lim in (ax.get_xlim(), ax.get_ylim(), ax.get_zlim())]
        ax.set_box_aspect(extent, zoom=1.25)
        ax.set_title("", loc="left")
        ax.patch.set_visible(False)
        name = SHORT_NAMES.get(code, ac.aircraft)
        fig.text(
            left + 0.012,
            bottom + 0.80 * cell_h,
            f"{code}  {name}",
            fontsize=10,
            fontweight="bold",
            color=INK,
        )
        fig.text(
            left + 0.012,
            bottom + 0.71 * cell_h,
            f"{ac.wing.span:.1f} m span",
            fontsize=8.5,
            color=INK2,
        )
    # The 3D axes reach past the figure edge, so a tight crop adds space.
    save(fig, "gallery.png", tight=False)


if __name__ == "__main__":
    fig_polar()
    fig_drag_rise()
    fig_breakdown()
    fig_fleet()
    fig_phases()
    fig_gallery()
