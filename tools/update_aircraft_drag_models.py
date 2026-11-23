#!/usr/bin/env python3
"""Generate aircraft drag-model configurations and coverage artifacts.

The tool replaces obsolete coefficient-only ``drag`` blocks with one uniform
physical-model block. Missing geometry is resolved only inside the model and
remains visible in the audit CSV and source sidecars.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from aerox.aircraft import load_aircraft_from_dict  # noqa: E402
from aerox.models.drag import DragBuildReport, build_drag_report  # noqa: E402
from aerox.models.drag.builder import resolve_drag_geometry  # noqa: E402
from tools.migrate_aircraft_data import (  # noqa: E402
    AIRCRAFT_ROOT,
    SOURCES_ROOT,
    dump_document,
    normalize_bundle,
    validate_simple,
)

REPORT_CSV = PROJECT_ROOT / "output/drag-model/drag-model-coverage.csv"
REPORT_YAML = PROJECT_ROOT / "output/drag-model/drag-model-coverage.yml"
REPORT_MD = PROJECT_ROOT / "output/drag-model/drag-model-coverage.md"
FIGURE_DIR = PROJECT_ROOT / "output/drag-model/figures"
FIGURE_PNG = FIGURE_DIR / "drag-model-coverage.png"
FIGURE_PDF = FIGURE_DIR / "drag-model-coverage.pdf"

SECTION_START = "DRAG MODEL"
SECTION_END = "END DRAG MODEL"
SUBTYPE_SECTION = "SUBTYPE AND ENGINE DATABANK APPLICABILITY\n"
SOURCE_ID = "aerox_geometry_drag_model_2026_08_08"

SOURCE_FIELDS = (
    "model",
    "configuration",
    "geometry_tier",
    "confidence",
    "reference_altitude_ft",
    "reference_mach",
    "reference_reynolds_per_m",
    "span_efficiency_factor",
    "unresolved_fraction_of_resolved_drag",
    "roughness_excrescence_fraction",
    "wave_drag_model",
    "wave_drag_source_tier",
    "wave_drag_confidence",
    "wave_drag_technology_factor",
)


def _rounded_parameters(parameters: DragBuildReport) -> dict[str, Any]:
    return {
        "model": parameters.model,
        "configuration": parameters.configuration,
        "geometry_tier": parameters.geometry_tier,
        "confidence": parameters.confidence,
        "reference_altitude_ft": round(parameters.reference_altitude_ft, 1),
        "reference_mach": round(parameters.reference_mach, 6),
        "reference_reynolds_per_m": round(parameters.reference_reynolds_per_m, 1),
        "span_efficiency_factor": round(parameters.span_efficiency_factor, 6),
        "unresolved_fraction_of_resolved_drag": _optional_round(
            parameters.unresolved_fraction_of_resolved_drag, 4
        ),
        "roughness_excrescence_fraction": _optional_round(
            parameters.roughness_excrescence_fraction, 4
        ),
        "wave_drag_model": parameters.wave_drag_model,
        "wave_drag_source_tier": parameters.wave_drag_source_tier,
        "wave_drag_confidence": parameters.wave_drag_confidence,
        "wave_drag_technology_factor": (
            None
            if parameters.wave_drag_technology_factor is None
            else round(parameters.wave_drag_technology_factor, 6)
        ),
    }


def _optional_round(value: float | None, digits: int) -> float | None:
    return None if value is None else round(value, digits)


def _source_section(parameters: DragBuildReport) -> str:
    values = _rounded_parameters(parameters)
    imputed = ", ".join(parameters.imputed_fields) or "none"
    lines = [
        SECTION_START,
        "  Active Aerox physical drag model and reference-condition polar snapshot.",
        f"  Scaled or substituted geometry fields: {imputed}",
        "",
    ]
    for field in SOURCE_FIELDS:
        value = values[field]
        lines.extend(
            [
                f"FIELD drag.{field}",
                f"  Value: {'null' if value is None else value}",
                "  Status: project-derived-active-model",
                f"  Source: {SOURCE_ID}",
            ]
        )
        if field == "model":
            if parameters.roughness_excrescence_fraction is None:
                allowance = (
                    "unresolved/resolved="
                    f"{parameters.unresolved_fraction_of_resolved_drag:.4f}"
                )
            else:
                allowance = (
                    "roughness_excrescence/computed="
                    f"{parameters.roughness_excrescence_fraction:.4f}; "
                    f"nacelle_tier={parameters.nacelle_geometry_tier}"
                )
            lines.append(
                "  Inputs: "
                f"configuration={parameters.configuration}; "
                f"tier={parameters.geometry_tier}; confidence={parameters.confidence}; "
                f"e_span={parameters.span_efficiency_factor:.4f}; "
                f"{allowance}"
            )
            terms = "; ".join(
                f"{term.name}={1.0e4 * term.value:.2f}"
                for term in parameters.parasite_terms
            )
            lines.append(f"  Zero-lift terms at reference (drag counts): {terms}")
            lines.append(
                "  Applicability: Active clean model; geometry-tier "
                "uncertainty applies."
            )
        elif field in {
            "configuration",
            "geometry_tier",
            "confidence",
            "wave_drag_model",
            "wave_drag_source_tier",
            "wave_drag_confidence",
        }:
            lines.append(
                "  Applicability: Model metadata; not a measured aircraft coefficient."
            )
        elif field == "reference_reynolds_per_m":
            lines.append(
                f"  Inputs: altitude={parameters.reference_altitude_ft:.1f} ft; "
                f"Mach={parameters.reference_mach:.6f}; ISA temperature and density"
            )
        elif field in {"reference_altitude_ft", "reference_mach"}:
            lines.append("  Applicability: Representative clean reference condition.")
        elif field in {
            "span_efficiency_factor",
            "unresolved_fraction_of_resolved_drag",
            "roughness_excrescence_fraction",
            "wave_drag_technology_factor",
        }:
            lines.append(
                "  Applicability: Physical model input; geometry-tier "
                "uncertainty applies."
            )
        lines.append("END FIELD")
    lines.extend([SECTION_END, ""])
    return "\n".join(lines)


def _replace_source_section(text: str, section: str) -> str:
    base = text
    for start, end in ((SECTION_START, SECTION_END),):
        if start in base:
            before, remainder = base.split(start, 1)
            if end not in remainder:
                raise ValueError("source note has an unterminated drag-model section")
            _, after = remainder.split(end, 1)
            base = before.rstrip() + "\n" + after.strip("\n") + "\n"
    if SUBTYPE_SECTION in base:
        before, after = base.split(SUBTYPE_SECTION, 1)
        return before.rstrip() + "\n\n" + section + "\n" + SUBTYPE_SECTION + after
    return base.rstrip() + "\n\n" + section


def _row(
    aircraft: Any,
    document: dict[str, Any],
    parameters: DragBuildReport,
    previous: dict[str, Any],
) -> dict[str, Any]:
    geometry = resolve_drag_geometry(aircraft)
    wave_zero_lift_critical_mach = None
    reference_wave_drag_at_cl_0_55 = None
    if parameters.wave_drag_technology_factor is not None:
        cosine = float(np.cos(np.deg2rad(geometry.wing_sweep_deg)))
        wave_zero_lift_critical_mach = (
            parameters.wave_drag_technology_factor / cosine
            - geometry.wing_thickness_to_chord_ratio / cosine**2
            - 0.108
        )
        critical = wave_zero_lift_critical_mach - 0.1 * 0.55 / cosine**3
        reference_wave_drag_at_cl_0_55 = (
            20.0 * max(parameters.reference_mach - critical, 0.0) ** 4
        )
    old = document.get("drag") or {}
    old_cd0 = previous.get("previous_cd0", previous.get("legacy_cd0", old.get("cd0")))
    old_k = previous.get("previous_k", previous.get("legacy_k", old.get("k")))
    return {
        "icao": document.get("icao"),
        "aircraft": document.get("aircraft"),
        "engine_type": (document.get("engine") or {}).get("type"),
        "geometry_tier": parameters.geometry_tier,
        "confidence": parameters.confidence,
        "imputed_parameter_count": len(parameters.imputed_fields),
        "imputed_fields": " | ".join(parameters.imputed_fields),
        "aspect_ratio": geometry.aspect_ratio,
        "previous_cd0": _optional_float(old_cd0),
        "active_cd0": parameters.cd0,
        "previous_k": _optional_float(old_k),
        "active_k": parameters.k,
        "reference_altitude_ft": parameters.reference_altitude_ft,
        "reference_mach": parameters.reference_mach,
        "reference_reynolds_per_m": (parameters.reference_reynolds_per_m),
        "span_efficiency_factor": parameters.span_efficiency_factor,
        "unresolved_fraction_of_resolved_drag": (
            parameters.unresolved_fraction_of_resolved_drag
        ),
        "roughness_excrescence_fraction": parameters.roughness_excrescence_fraction,
        "nacelle_geometry_tier": parameters.nacelle_geometry_tier,
        **{f"cd0_{term.name}": term.value for term in parameters.parasite_terms},
        "wave_drag_model": parameters.wave_drag_model,
        "wave_drag_source_tier": parameters.wave_drag_source_tier,
        "wave_drag_confidence": parameters.wave_drag_confidence,
        "wave_drag_technology_factor": parameters.wave_drag_technology_factor,
        "wave_zero_lift_critical_mach": wave_zero_lift_critical_mach,
        "reference_wave_drag_at_cl_0_55": reference_wave_drag_at_cl_0_55,
    }


def _optional_float(value: Any) -> float | None:
    return None if value in {None, ""} else float(value)


def _previous_rows() -> dict[str, dict[str, Any]]:
    if not REPORT_CSV.exists():
        return {}
    with REPORT_CSV.open(encoding="utf-8", newline="") as stream:
        return {str(row["icao"]): row for row in csv.DictReader(stream)}


def _write_csv(rows: list[dict[str, Any]]) -> None:
    REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_CSV.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tiers = Counter(str(row["geometry_tier"]) for row in rows)
    engines = Counter(str(row["engine_type"]) for row in rows)
    previous = [row for row in rows if row["previous_cd0"] is not None]
    wave = [row for row in rows if row["wave_drag_model"] is not None]
    return {
        "schema_version": 1,
        "status": "active-aircraft-geometry-drag-model",
        "model": "aerox-geometry-drag-v2",
        "aircraft_total": len(rows),
        "previous_polar_coverage": len(previous),
        "active_model_coverage": len(rows),
        "active_wave_model_coverage": len(wave),
        "engine_counts": dict(sorted(engines.items())),
        "geometry_tier_counts": dict(sorted(tiers.items())),
        "reference_condition": (
            "class-specific representative cruise altitude and recorded cruise Mach/TAS"
        ),
        "activation_policy": "one active drag block; no legacy fallback",
        "coefficient_ranges": {
            field: {
                "minimum": float(
                    min(row[field] for row in rows if row[field] is not None)
                ),
                "median": float(
                    np.median([row[field] for row in rows if row[field] is not None])
                ),
                "maximum": float(
                    max(row[field] for row in rows if row[field] is not None)
                ),
            }
            for field in ("active_cd0", "active_k")
        },
        "wave_reference_diagnostic": {
            "lift_coefficient": 0.55,
            "minimum_drag_counts": float(
                1.0e4 * min(row["reference_wave_drag_at_cl_0_55"] for row in wave)
            ),
            "median_drag_counts": float(
                1.0e4
                * np.median([row["reference_wave_drag_at_cl_0_55"] for row in wave])
            ),
            "maximum_drag_counts": float(
                1.0e4 * max(row["reference_wave_drag_at_cl_0_55"] for row in wave)
            ),
        },
        "limitations": [
            "database-wide values remain physics-based estimates, not type-specific "
            "validation results",
            "scaled turboprop, piston, and special-aircraft geometry has low "
            "confidence",
            "the parabolic snapshot excludes transonic wave drag by design",
            "wave technology is a low-confidence class prior except where "
            "aircraft-specific public evidence becomes available",
            "deployed-configuration drag increments are not yet modeled",
        ],
        "artifacts": {
            "rows": str(REPORT_CSV.relative_to(PROJECT_ROOT)),
            "report": str(REPORT_MD.relative_to(PROJECT_ROOT)),
            "figure_png": str(FIGURE_PNG.relative_to(PROJECT_ROOT)),
            "figure_pdf": str(FIGURE_PDF.relative_to(PROJECT_ROOT)),
        },
    }


def _write_markdown(summary: dict[str, Any]) -> None:
    tiers = summary["geometry_tier_counts"]
    ranges = summary["coefficient_ranges"]
    tier_lines = "\n".join(f"| {name} | {count} |" for name, count in tiers.items())
    text = f"""# Aircraft drag model coverage

Aerox now carries one uniform active physical drag model for all
{summary["aircraft_total"]} bundled aircraft. The former coefficient-only
implementation and parallel `drag_prior` block have been removed. The active
`drag` block contains only independent model parameters. Reference-condition
`CD0` and `k` are derived for audit and low-order adapters and are not stored.

## Formulation

At each aircraft's documented reference condition, a clean polar projection is

`CD = CD0 + k CL^2`,

with component parasite drag from Reynolds-dependent skin friction, form
factors, and wetted geometry. Civil turbofans use a component build-up:
exposed wing, fuselage, tails with interference factor 1.04, nacelles with
Raymer form and mount-dependent interference factors, pylons at 20 % of the
nacelle drag, Kroo upsweep, gap, and nacelle-base terms, and a published
roughness and excrescence allowance of 9 % of the computed sum. Turboprop,
piston, and high-speed aircraft keep one lumped unresolved allowance. The
effective quadratic coefficient combines span-induced and lift-dependent
profile drag. `CD0`, `k`, and the old effective Oswald
`e = 1 / (pi AR k)` are therefore derived, not independent parameters, and are
not stored. `span_efficiency_factor` remains because it is the physical input
to the pure span-induced term.

For civil turbofans, the span-only term uses the Poll–Schumann geometry branch
`e_span = 1 / (1.03 + 2 (b_f / b)^2)`. Aerox then adds the separate
lift-profile contribution `0.80 (1 - 0.53 cos(sweep)) CD0`. This separation is
intentional: using a conventional effective Oswald factor as `e_span` would
double-count lift-dependent profile drag. Niţă–Scholz remains the preferred
planform branch once trustworthy taper ratios are available; none of the 145
current YAML records has a published taper ratio.

Civil-turbofan reference pressure is solved from clean lift at 0.8 MTOW,
recorded cruise Mach, and `CL=0.556`. Piston and turboprop reference conditions
use recorded cruise TAS/Mach and a class-scaled fraction of ceiling. Every
result stores altitude, Mach, and unit Reynolds number, so `CD0` has an explicit
condition rather than an implicit common-altitude assumption. A sparse legacy
cruise-Mach value is capped at the aircraft's published MMO before the
reference condition is solved.

Missing geometry is scaled with frozen class ratios. It is never written back
as certified geometry: every substituted field is listed in
[{REPORT_CSV.name}]({REPORT_CSV.name}).

## Coverage and confidence

| Geometry tier | Aircraft |
|---|---:|
{tier_lines}

Coverage increases from {summary["previous_polar_coverage"]} previous polars to
{summary["active_model_coverage"]} active physical models. Coefficient ranges
are `CD0={ranges["active_cd0"]["minimum"]:.4f}–{ranges["active_cd0"]["maximum"]:.4f}`
and `k={ranges["active_k"]["minimum"]:.4f}–{ranges["active_k"]["maximum"]:.4f}`.

## Interpretation

This is a better *aircraft model system*: it is internally consistent, condition
referenced, provenance tracked, and complete. It is not evidence that all 145
aircraft are now more accurate in absolute terms. Tier A/B civil transports
have the strongest geometry support. Turboprops, piston aircraft, and special
military layouts remain low-confidence until public geometry or flight data
replaces the scaled defaults.

The low-order snapshot is subcritical and parabolic. Runtime Reynolds variation
is handled by the condition-aware physical model. For
{summary["active_wave_model_coverage"]} subsonic civil turbofans, the runtime
model also derives critical Mach from sweep, thickness ratio, lift coefficient,
and a stored airfoil-technology factor, then applies the Lock fourth-power wave
increment. The coefficient `20` and the critical-Mach equation are owned by the
versioned `aerox-korn-lock-wave-v1` formulation rather than duplicated in every
aircraft file. Piston, turboprop, supersonic, and special military layouts are
explicitly marked not applicable; no transonic prior is fabricated for them.
At each stored reference Mach and `CL=0.55`, the 90 enabled priors span
`{summary["wave_reference_diagnostic"]["minimum_drag_counts"]:.2f}` to
`{summary["wave_reference_diagnostic"]["maximum_drag_counts"]:.2f}` drag counts
(median `{summary["wave_reference_diagnostic"]["median_drag_counts"]:.2f}`).
These are audit outputs, not additional YAML parameters. Large values occur at
or near MMO and should be replaced by type-specific evidence before claiming
aircraft-deck accuracy.

![Aircraft drag-model comparison and coverage](figures/drag-model-coverage.png)

## Reproduction

Run `uv run python tools/update_aircraft_drag_models.py --write` to regenerate
the YAML drag blocks, source-note provenance, table, and comparison figure.
Use `--check` in CI to detect drift.
"""
    REPORT_MD.write_text(text, encoding="utf-8")


def _write_figure(rows: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/aerox-matplotlib")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
        }
    )
    colors = {"turbofan": "#0072B2", "turboprop": "#E69F00", "piston": "#009E73"}
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)

    for ax, old_key, new_key, label in (
        (axes[0, 0], "previous_cd0", "active_cd0", r"$C_{D0}$"),
        (axes[0, 1], "previous_k", "active_k", r"effective $k$"),
    ):
        comparable = [row for row in rows if row[old_key] is not None]
        for engine, color in colors.items():
            subset = [row for row in comparable if row["engine_type"] == engine]
            if subset:
                ax.scatter(
                    [row[old_key] for row in subset],
                    [row[new_key] for row in subset],
                    s=25,
                    alpha=0.78,
                    color=color,
                    edgecolor="white",
                    linewidth=0.35,
                    label=engine,
                )
        values = [row[key] for row in comparable for key in (old_key, new_key)]
        if not values:
            ax.set_visible(False)
            continue
        low, high = min(values), max(values)
        margin = 0.06 * (high - low)
        ax.plot(
            [low - margin, high + margin],
            [low - margin, high + margin],
            "--",
            color="0.35",
            lw=1,
        )
        ax.set_xlim(low - margin, high + margin)
        ax.set_ylim(low - margin, high + margin)
        ax.set_xlabel(f"Previous database {label}")
        ax.set_ylabel(f"Active physical model {label}")
        ax.grid(alpha=0.25)
    axes[0, 0].set_title(
        "a  Zero-lift drag: previous vs active", loc="left", fontweight="bold"
    )
    axes[0, 1].set_title(
        "b  Quadratic drag: previous vs active", loc="left", fontweight="bold"
    )
    axes[0, 0].legend(frameon=False)

    tier_counts = Counter(row["geometry_tier"] for row in rows)
    tier_order = sorted(tier_counts)
    axes[1, 0].barh(
        [tier.replace("-", " ") for tier in tier_order],
        [tier_counts[tier] for tier in tier_order],
        color="#56B4E9",
    )
    for index, tier in enumerate(tier_order):
        axes[1, 0].text(
            tier_counts[tier] + 1,
            index,
            str(tier_counts[tier]),
            va="center",
        )
    axes[1, 0].set_xlabel("Aircraft count")
    axes[1, 0].set_title("c  Geometry support tier", loc="left", fontweight="bold")
    axes[1, 0].grid(axis="x", alpha=0.25)

    wave_rows = [row for row in rows if row["wave_drag_model"] is not None]
    axes[1, 1].scatter(
        [row["reference_mach"] for row in wave_rows],
        [1.0e4 * row["reference_wave_drag_at_cl_0_55"] for row in wave_rows],
        s=28,
        alpha=0.78,
        color="#009E73",
        edgecolor="white",
        linewidth=0.35,
        label="subsonic civil turbofan",
    )
    annotation_offsets = ((4, 4), (4, 4), (4, 7), (4, -11))
    for row, offset in zip(
        sorted(
            wave_rows,
            key=lambda item: item["reference_wave_drag_at_cl_0_55"],
            reverse=True,
        )[:4],
        annotation_offsets,
        strict=True,
    ):
        axes[1, 1].annotate(
            row["icao"],
            (
                row["reference_mach"],
                1.0e4 * row["reference_wave_drag_at_cl_0_55"],
            ),
            xytext=offset,
            textcoords="offset points",
            fontsize=7,
        )
    axes[1, 1].set_xlabel("Reference Mach number")
    axes[1, 1].set_ylabel(r"Wave increment at $C_L=0.55$ (drag counts)")
    axes[1, 1].set_title("d  Wave-model diagnostic", loc="left", fontweight="bold")
    axes[1, 1].grid(alpha=0.25)
    axes[1, 1].legend(frameon=False)

    fig.suptitle(
        "Aerox aircraft drag models — physical consistency and geometry coverage",
        fontsize=12,
        fontweight="bold",
    )
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PNG, bbox_inches="tight")
    fig.savefig(FIGURE_PDF, bbox_inches="tight")
    plt.close(fig)


def _write_artifacts(rows: list[dict[str, Any]]) -> None:
    rows.sort(key=lambda row: str(row["icao"]))
    _write_csv(rows)
    summary = _summary(rows)
    REPORT_YAML.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    _write_markdown(summary)
    _write_figure(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=AIRCRAFT_ROOT)
    parser.add_argument("--icao", action="append", default=[])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected = {code.upper() for code in args.icao}
    root = args.root
    sources_root = (
        SOURCES_ROOT
        if root.resolve() == AIRCRAFT_ROOT.resolve()
        else root.parent / "sources"
    )
    files = [
        path
        for path in sorted(root.glob("*/*.yml"))
        if not selected or path.stem.upper() in selected
    ]
    changes: list[tuple[Path, str, Path, str]] = []
    rows: list[dict[str, Any]] = []
    previous_rows = _previous_rows()
    for path in files:
        original = path.read_text(encoding="utf-8")
        document = yaml.safe_load(original)
        sidecar_path = sources_root / path.parent.name / f"{path.stem}.txt"
        old_sidecar = sidecar_path.read_text(encoding="utf-8")
        generation_document = dict(document)
        generation_document["drag"] = None
        aircraft = load_aircraft_from_dict(generation_document)
        parameters = build_drag_report(aircraft)
        normalized, _ = normalize_bundle(document, old_sidecar)
        normalized["drag"] = _rounded_parameters(parameters)
        issues = validate_simple(normalized)
        if issues:
            raise ValueError(f"{path}: " + "; ".join(issues))
        rendered = dump_document(normalized)
        sidecar = _replace_source_section(old_sidecar, _source_section(parameters))
        if rendered != original or sidecar != old_sidecar:
            changes.append((path, rendered, sidecar_path, sidecar))
        rows.append(
            _row(
                aircraft,
                document,
                parameters,
                previous_rows.get(path.stem.upper(), {}),
            )
        )

    if args.write:
        for path, rendered, sidecar_path, sidecar in changes:
            path.write_text(rendered, encoding="utf-8")
            sidecar_path.write_text(sidecar, encoding="utf-8")
        if root.resolve() == AIRCRAFT_ROOT.resolve() and not selected:
            _write_artifacts(rows)
    mode = "write" if args.write else "check"
    print(f"aircraft={len(files)} changed={len(changes)} mode={mode}")
    return 1 if args.check and changes else 0


if __name__ == "__main__":
    raise SystemExit(main())
