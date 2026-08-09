#!/usr/bin/env python3
"""Build and run the CAD-derived A320 thin-surface VSPAERO prior.

This tool intentionally generates a symmetric-airfoil VLM stability model. It
is an independent geometry-based derivative prior, not an A320 aerodynamic
database and not a replacement for Aerox's viscous drag model. VSPAERO's saved
total-CD field includes its configured ``CDo`` as well as ``CDi`` and must not
be described as a pure induced-drag result.

OpenVSP's Linux bundle ships Python bindings beside its executables.  When the
bindings are not installed into the active environment, this script discovers
them from the ``vspaero`` executable and re-executes itself with the required
Python and shared-library search paths.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import ctypes
import math
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.build_a320_geometry_mass_prior import (
    FUSELAGE_WIDTH_M,
    HORIZONTAL_TAIL_SPAN_M,
    WING_MAC_M,
    WING_SPAN_M,
    _nominal_geometry,
    load_measurements,
)

UPDATED = "2026-08-05"
SREF_M2 = 122.0
CG_FRACTION_MAC = 0.265
MACH_CASES = (0.20, 0.50, 0.78)
ALPHA_DEG = 2.0
MESHES = {
    "coarse": {"wing_u": 8, "wing_w": 17, "tail_u": 10, "tail_w": 13},
    "medium": {"wing_u": 14, "wing_w": 25, "tail_u": 18, "tail_w": 19},
    "fine": {"wing_u": 20, "wing_w": 37, "tail_u": 26, "tail_w": 29},
}
DERIVATIVE_COLUMNS = {
    "CL_alpha": ("CL", "alpha"),
    "CMm_alpha": ("CMm", "alpha"),
    "CMm_q": ("CMm", "q"),
    "CS_beta": ("CS", "beta"),
    "CMl_beta": ("CMl", "beta"),
    "CMl_p": ("CMl", "p"),
    "CMl_r": ("CMl", "r"),
    "CMn_beta": ("CMn", "beta"),
    "CMn_p": ("CMn", "p"),
    "CMn_r": ("CMn", "r"),
}
CONVERGENCE_COLUMNS = (
    "CL_alpha",
    "CMm_alpha",
    "CMm_q",
    "CS_beta",
    "CMl_beta",
    "CMl_p",
    "CMn_beta",
    "CMn_r",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements",
        type=Path,
        default=Path("docs/modeling/a320-cad-geometry-measurements.csv"),
    )
    parser.add_argument(
        "--geometry-prior",
        type=Path,
        default=Path("docs/modeling/a320-geometry-mass-prior.yml"),
    )
    parser.add_argument(
        "--parameters", type=Path, default=Path("docs/modeling/a320-vspaero-prior.yml")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("docs/modeling/a320-vspaero-prior.csv")
    )
    parser.add_argument(
        "--convergence",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-convergence.csv"),
    )
    parser.add_argument(
        "--report", type=Path, default=Path("docs/modeling/a320-vspaero-prior.md")
    )
    parser.add_argument(
        "--figure-prefix",
        type=Path,
        default=Path("docs/modeling/figures/a320-vspaero-prior"),
    )
    parser.add_argument(
        "--model-output",
        type=Path,
        default=Path("docs/modeling/a320-vspaero-nominal.vsp3"),
    )
    parser.add_argument("--openvsp-root", type=Path)
    parser.add_argument("--jobs", type=int, default=min(32, os.cpu_count() or 4))
    parser.add_argument("--keep-workdir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def _openvsp_root(explicit: Path | None = None) -> Path:
    if explicit:
        return explicit.resolve()
    executable = shutil.which("vspaero")
    if not executable:
        raise RuntimeError("vspaero is not on PATH; install OpenVSP or pass --openvsp-root")
    resolved = Path(executable).resolve()
    return resolved.parent


def _bootstrap_openvsp(root: Path) -> None:
    """Re-exec once so bundled Python bindings can load their libpython."""

    if os.environ.get("AEROX_OPENVSP_BOOTSTRAPPED") == "1":
        return
    python_root = root / "python"
    package_roots = [python_root / "openvsp", python_root / "degen_geom", python_root / "utilities"]
    if not (package_roots[0] / "openvsp" / "_vsp.so").exists():
        raise RuntimeError(f"OpenVSP Python bindings were not found below {python_root}")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        [*(str(path) for path in package_roots), existing_pythonpath]
    ).rstrip(os.pathsep)
    libdir = str(sysconfig.get_config_var("LIBDIR") or "")
    existing_ldpath = env.get("LD_LIBRARY_PATH", "")
    env["LD_LIBRARY_PATH"] = os.pathsep.join([libdir, existing_ldpath]).rstrip(os.pathsep)
    env["AEROX_OPENVSP_BOOTSTRAPPED"] = "1"
    os.execve(sys.executable, [sys.executable, *sys.argv], env)


@contextlib.contextmanager
def _redirect_process_output(path: Path, enabled: bool = True) -> Iterator[None]:
    """Redirect Python and C/C++ stdout/stderr during noisy VSPAERO runs."""

    if not enabled:
        yield
        return
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        ctypes.CDLL(None).fflush(None)
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        os.close(saved[0])
        os.close(saved[1])
        os.close(fd)


def _wing_planform(measurements: dict[str, Any]) -> tuple[list[float], list[float], list[float]]:
    left_tail = measurements["horizontal_tail_leading_edge_left"]
    right_tail = measurements["horizontal_tail_leading_edge_right"]
    center_x = (left_tail.p1[0] + right_tail.p1[0]) / 2.0
    nose_y = min(
        measurements["top_view_nose_left"].p1[1],
        measurements["top_view_nose_right"].p1[1],
    )
    le_inner = measurements["wing_leading_edge_inboard_right"]
    le_outer = measurements["wing_leading_edge_outboard_right"]
    te_inner = measurements["wing_trailing_edge_inboard_right"]
    te_outer = measurements["wing_trailing_edge_outboard_right"]
    semispan = [
        0.0,
        FUSELAGE_WIDTH_M / 2.0,
        (le_outer.p2[0] - center_x) / 1000.0,
        (le_outer.p1[0] - center_x) / 1000.0,
        WING_SPAN_M / 2.0,
    ]
    leading_edge = [
        (le_inner.p2[1] - nose_y) / 1000.0,
        (le_inner.p2[1] - nose_y) / 1000.0,
        (le_outer.p2[1] - nose_y) / 1000.0,
        (le_outer.p1[1] - nose_y) / 1000.0,
        (le_outer.p1[1] + 817.0 - nose_y) / 1000.0,
    ]
    trailing_edge = [
        (te_inner.p1[1] - nose_y) / 1000.0,
        (te_inner.p1[1] - nose_y) / 1000.0,
        (te_outer.p1[1] - nose_y) / 1000.0,
        (te_outer.p2[1] - nose_y) / 1000.0,
        (le_outer.p1[1] + 817.0 - nose_y) / 1000.0,
    ]
    chord = [te - le for le, te in zip(leading_edge, trailing_edge, strict=True)]
    chord[-1] = 0.05  # avoid a singular numerical point at the rounded/fenced tip
    return semispan, leading_edge, chord


def _tail_root_le(ac_station: float, full_span: float, root: float, tip: float, sweep_q: float) -> float:
    taper = tip / root
    y_mac = (full_span / 6.0) * (1.0 + 2.0 * taper) / (1.0 + taper)
    mac = (2.0 / 3.0) * root * (1.0 + taper + taper**2) / (1.0 + taper)
    tan_le = math.tan(math.radians(sweep_q)) + 0.25 * (root - tip) / (full_span / 2.0)
    return ac_station - tan_le * y_mac - 0.25 * mac


def _add_wing(
    vsp: Any,
    name: str,
    semispan: list[float],
    leading_edge: list[float],
    chord: list[float],
    dihedral: list[float],
    root_z: float,
    tess_u: int,
    tess_w: int,
    *,
    symmetric: bool,
    x_rotation: float = 0.0,
    thickness_ratio: float = 0.12,
) -> str:
    geom_id = vsp.AddGeom("WING", "")
    vsp.SetGeomName(geom_id, name)
    segment_count = len(semispan) - 1
    for _ in range(segment_count):
        vsp.InsertXSec(geom_id, 1, vsp.XS_FOUR_SERIES)
    vsp.CutXSec(geom_id, 1)
    vsp.SetParmVal(
        geom_id, "Sym_Planar_Flag", "Sym", vsp.SYM_XZ if symmetric else vsp.SYM_NONE
    )
    vsp.SetParmVal(geom_id, "X_Rel_Location", "XForm", leading_edge[0])
    vsp.SetParmVal(geom_id, "Z_Rel_Location", "XForm", root_z)
    vsp.SetParmVal(geom_id, "X_Rel_Rotation", "XForm", x_rotation)
    vsp.SetParmVal(geom_id, "Tess_W", "Shape", tess_w)
    for index in range(1, segment_count + 1):
        projected_span = semispan[index] - semispan[index - 1]
        dx = leading_edge[index] - leading_edge[index - 1]
        section_dihedral = dihedral[index - 1]
        # OpenVSP's Span driver follows the tilted section. The manufacturer
        # span is a front/top-view projection, so lengthen the section and
        # compute sweep against that length to preserve both projected span
        # and the CAD leading-edge station.
        span = projected_span / math.cos(math.radians(section_dihedral))
        vsp.SetDriverGroup(
            geom_id,
            index,
            vsp.SPAN_WSECT_DRIVER,
            vsp.ROOTC_WSECT_DRIVER,
            vsp.TIPC_WSECT_DRIVER,
        )
        group = f"XSec_{index}"
        vsp.SetParmVal(geom_id, "Span", group, span)
        vsp.SetParmVal(geom_id, "Root_Chord", group, chord[index - 1])
        vsp.SetParmVal(geom_id, "Tip_Chord", group, chord[index])
        vsp.SetParmVal(geom_id, "Sweep_Location", group, 0.0)
        vsp.SetParmVal(geom_id, "Sweep", group, math.degrees(math.atan2(dx, span)))
        vsp.SetParmVal(geom_id, "Dihedral", group, section_dihedral)
        vsp.SetParmVal(geom_id, "SectTess_U", group, tess_u)
        vsp.Update()
    xsec_surface = vsp.GetXSecSurf(geom_id, 0)
    for index in range(vsp.GetNumXSec(xsec_surface)):
        vsp.ChangeXSecShape(xsec_surface, index, vsp.XS_FOUR_SERIES)
        xsec_id = vsp.GetXSec(xsec_surface, index)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, "Camber"), 0.0)
        vsp.SetParmVal(vsp.GetXSecParm(xsec_id, "ThickChord"), thickness_ratio)
    vsp.Update()
    return geom_id


def _build_model(vsp: Any, model_path: Path, mesh: dict[str, int], geometry: dict[str, Any], measurements: dict[str, Any]) -> None:
    vsp.ClearVSPModel()
    y, le, chord = _wing_planform(measurements)
    wing_dihedral = geometry["wing_reference"]["exposed_dihedral_deg"]
    wing_id = _add_wing(
        vsp,
        "A320_Main_Wing",
        y,
        le,
        chord,
        [0.0, wing_dihedral, wing_dihedral, wing_dihedral],
        0.0,
        mesh["wing_u"],
        mesh["wing_w"],
        symmetric=True,
    )

    horizontal = geometry["horizontal_tail"]
    h_span = horizontal["span_m"]
    h_root = horizontal["root_chord_m"]
    h_tip = horizontal["tip_chord_m"]
    h_root_le = _tail_root_le(
        horizontal["aerodynamic_center_station_from_nose_m"],
        h_span,
        h_root,
        h_tip,
        horizontal["quarter_chord_sweep_deg"],
    )
    h_le_tip = h_root_le + (
        math.tan(math.radians(horizontal["quarter_chord_sweep_deg"]))
        + 0.25 * (h_root - h_tip) / (h_span / 2.0)
    ) * (h_span / 2.0)
    _add_wing(
        vsp,
        "A320_Horizontal_Tail",
        [0.0, h_span / 2.0],
        [h_root_le, h_le_tip],
        [h_root, h_tip],
        [horizontal["dihedral_deg"]],
        horizontal["root_vertical_offset_from_wing_m"],
        mesh["tail_u"],
        mesh["tail_w"],
        symmetric=True,
        thickness_ratio=0.10,
    )

    vertical = geometry["vertical_tail"]
    v_height = vertical["reference_height_m"]
    v_root = vertical["root_chord_m"]
    v_tip = vertical["tip_chord_m"]
    # The tail helper expects a full symmetric span; the vertical tail's height
    # is one half of that mathematical reference span.
    v_root_le = _tail_root_le(
        vertical["aerodynamic_center_station_from_nose_m"],
        2.0 * v_height,
        v_root,
        v_tip,
        vertical["quarter_chord_sweep_deg"],
    )
    v_le_tip = v_root_le + (
        math.tan(math.radians(vertical["quarter_chord_sweep_deg"]))
        + 0.25 * (v_root - v_tip) / v_height
    ) * v_height
    _add_wing(
        vsp,
        "A320_Vertical_Tail",
        [0.0, v_height],
        [v_root_le, v_le_tip],
        [v_root, v_tip],
        [0.0],
        vertical["root_vertical_offset_from_wing_m"],
        mesh["tail_u"],
        mesh["tail_w"],
        symmetric=False,
        x_rotation=90.0,
        thickness_ratio=0.10,
    )
    vsp.Update()
    vsp.SetVSPAERORefWingID(wing_id)
    vsp.WriteVSPFile(str(model_path), vsp.SET_ALL)


def _set_int(vsp: Any, analysis: str, name: str, value: int) -> None:
    vsp.SetIntAnalysisInput(analysis, name, [int(value)])


def _set_double(vsp: Any, analysis: str, name: str, value: float) -> None:
    vsp.SetDoubleAnalysisInput(analysis, name, [float(value)])


def parse_stability_file(path: Path) -> dict[str, float]:
    """Parse base coefficients and dimensional-rate stability derivatives."""

    lines = path.read_text().splitlines()
    derivative_header = next(
        index for index, line in enumerate(lines) if line.startswith("Coef") and "Total" in line
    )
    table: dict[str, dict[str, float]] = {}
    labels = ("base", "alpha", "beta", "p", "q", "r", "mach", "u")
    for line in lines[derivative_header + 1 :]:
        fields = line.split()
        if len(fields) != 9:
            if table and line.startswith("#"):
                break
            continue
        try:
            values = [float(value) for value in fields[1:]]
        except ValueError:
            continue
        table[fields[0]] = dict(zip(labels, values, strict=True))
    if not {"CL", "CMm", "CS", "CMl", "CMn"}.issubset(table):
        raise ValueError(f"Incomplete VSPAERO stability table in {path}")
    result = {
        "CL": table["CL"]["base"],
        "CD_solver_total": table["CD"]["base"],
        "CS": table["CS"]["base"],
        "CMl": table["CMl"]["base"],
        "CMm": table["CMm"]["base"],
        "CMn": table["CMn"]["base"],
    }
    result.update(
        {name: table[coefficient][variable] for name, (coefficient, variable) in DERIVATIVE_COLUMNS.items()}
    )
    static_margin_line = next(line for line in lines if line.startswith("SM"))
    result["static_margin"] = float(static_margin_line.split()[1])
    return result


def _mesh_counts(log_path: Path) -> dict[str, int]:
    text = log_path.read_text(errors="replace")
    loops = re.findall(r"Number Of Vortex Loop Equations:\s+(\d+)", text)
    nodes = re.findall(r"NumNodes:\s+(\d+)", text)
    return {
        "vortex_loop_equations": int(loops[-1]) if loops else -1,
        "geometry_nodes": int(nodes[-1]) if nodes else -1,
    }


def _run_case(
    vsp: Any,
    case_dir: Path,
    mesh_name: str,
    mach: float,
    geometry: dict[str, Any],
    measurements: dict[str, Any],
    jobs: int,
    verbose: bool,
    alpha_deg: float = ALPHA_DEG,
) -> tuple[dict[str, Any], Path]:
    case_dir.mkdir(parents=True, exist_ok=True)
    alpha_tag = f"{alpha_deg:+05.1f}".replace("+", "p").replace("-", "m").replace(".", "d")
    base = f"a320_{mesh_name}_m{round(mach * 100):02d}_a{alpha_tag}"
    model_path = case_dir / f"{base}.vsp3"
    log_path = case_dir / f"{base}.log"
    with _redirect_process_output(log_path, enabled=not verbose):
        _build_model(vsp, model_path, MESHES[mesh_name], geometry, measurements)
        analysis = "VSPAEROComputeGeometry"
        vsp.SetAnalysisInputDefaults(analysis)
        _set_int(vsp, analysis, "GeomSet", vsp.SET_NONE)
        _set_int(vsp, analysis, "ThinGeomSet", vsp.SET_ALL)
        vsp.ExecAnalysis(analysis)

        analysis = "VSPAEROSweep"
        vsp.SetAnalysisInputDefaults(analysis)
        _set_int(vsp, analysis, "GeomSet", vsp.SET_NONE)
        _set_int(vsp, analysis, "ThinGeomSet", vsp.SET_ALL)
        _set_int(vsp, analysis, "RefFlag", 0)
        _set_double(vsp, analysis, "Sref", SREF_M2)
        _set_double(vsp, analysis, "bref", WING_SPAN_M)
        _set_double(vsp, analysis, "cref", WING_MAC_M)
        _set_double(vsp, analysis, "Xcg", geometry["wing_reference"]["reference_cg_station_from_nose_m"])
        _set_double(vsp, analysis, "Ycg", 0.0)
        _set_double(vsp, analysis, "Zcg", 0.0)
        _set_double(vsp, analysis, "MachStart", mach)
        _set_double(vsp, analysis, "MachEnd", mach)
        _set_int(vsp, analysis, "MachNpts", 1)
        _set_double(vsp, analysis, "AlphaStart", alpha_deg)
        _set_double(vsp, analysis, "AlphaEnd", alpha_deg)
        _set_int(vsp, analysis, "AlphaNpts", 1)
        _set_double(vsp, analysis, "BetaStart", 0.0)
        _set_double(vsp, analysis, "BetaEnd", 0.0)
        _set_int(vsp, analysis, "BetaNpts", 1)
        _set_int(vsp, analysis, "NCPU", jobs)
        _set_int(vsp, analysis, "WakeNumIter", 5)
        _set_int(vsp, analysis, "UnsteadyType", vsp.STABILITY_DEFAULT)
        vsp.ExecAnalysis(analysis)
    stab_path = case_dir / f"{base}.stab"
    if not stab_path.exists():
        raise RuntimeError(f"VSPAERO did not produce {stab_path}; inspect {log_path}")
    row: dict[str, Any] = {
        "mesh": mesh_name,
        "mach": mach,
        "alpha_deg": alpha_deg,
        **_mesh_counts(log_path),
        **parse_stability_file(stab_path),
    }
    return row, model_path


def _relative_difference(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1e-9)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot(results: list[dict[str, Any]], convergence: list[dict[str, Any]], prefix: Path) -> None:
    import matplotlib.pyplot as plt

    colors = {"coarse": "#D55E00", "medium": "#0072B2", "fine": "#009E73"}
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.2), constrained_layout=True)
    labels = [row["derivative"].replace("_", "\n") for row in convergence]
    values = [100.0 * row["medium_fine_relative_difference"] for row in convergence]
    axes[0].bar(labels, values, color="#0072B2", alpha=0.85)
    axes[0].axhline(5.0, color="#A33A2B", linestyle="--", linewidth=1.2, label="5% gate")
    axes[0].set(title="Mesh convergence at M=0.50", ylabel="medium–fine difference [%]")
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False)

    medium = sorted(
        (row for row in results if row["mesh"] == "medium"), key=lambda row: row["mach"]
    )
    mach = [row["mach"] for row in medium]
    for name, marker in (("CL_alpha", "o"), ("CMm_alpha", "s")):
        axes[1].plot(mach, [row[name] for row in medium], marker=marker, label=name)
    axes[1].set(title="Longitudinal geometry prior", xlabel="Mach", ylabel="derivative [per rad]")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    for name, marker in (("CS_beta", "o"), ("CMl_beta", "s"), ("CMn_beta", "^")):
        axes[2].plot(mach, [row[name] for row in medium], marker=marker, label=name)
    axes[2].set(title="Lateral geometry prior", xlabel="Mach", ylabel="derivative [per rad]")
    axes[2].grid(alpha=0.25)
    axes[2].legend(frameon=False)
    fig.suptitle(
        "A320 CAD-planform VSPAERO cross-check — thin-surface VLM, not flight-identified",
        fontsize=11,
    )
    prefix.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(prefix.with_suffix(".png"), dpi=220)
    fig.savefig(prefix.with_suffix(".pdf"))
    plt.close(fig)


def _render_report(prior: dict[str, Any]) -> str:
    nominal = prior["nominal_medium_mesh"]
    gate = prior["mesh_convergence_gate"]
    return f"""# A320 VSPAERO geometry-based derivative prior

## Outcome

OpenVSP {prior['software']['openvsp_version']} and VSPAERO
{prior['software']['vspaero_version']} successfully ran a thin-surface A320
wing-and-tail model reconstructed from the Airbus May 2012 wing-tip-fence CAD.
The medium-to-fine differences are at most
{100 * gate['maximum_medium_fine_relative_difference']:.2f}% across the eight
gated derivatives, so the numerical mesh gate is **{'passed' if gate['passed'] else 'not passed'}**.

At Mach 0.50 and 2 deg angle of attack, the medium mesh gives:

| Quantity | VSPAERO value |
|---|---:|
| CL_alpha | {nominal['CL_alpha']:.4f} per rad |
| CMm_alpha | {nominal['CMm_alpha']:.4f} per rad |
| CMm_q | {nominal['CMm_q']:.4f} per rad of dimensional pitch rate at the recorded reference speed |
| CS_beta | {nominal['CS_beta']:.4f} per rad |
| CMl_beta | {nominal['CMl_beta']:.4f} per rad |
| CMl_p | {nominal['CMl_p']:.4f} per rad of dimensional roll rate at the recorded reference speed |
| CMn_beta | {nominal['CMn_beta']:.4f} per rad |
| CMn_r | {nominal['CMn_r']:.4f} per rad of dimensional yaw rate at the recorded reference speed |

## What this model contains

- The certified/reference 122 m2 wing area, 34.1 m span, and 4.1935 m MAC.
- CAD-derived piecewise wing planform, 5.29 deg exposed-wing dihedral, tail
  planforms, tail vertical placement, and the Airbus 26.5%-MAC planning CG.
- Symmetric four-series mean surfaces. The actual A320 proprietary airfoil loft,
  twist, aeroelastic deformation, fuselage, nacelles, pylons, high-lift devices,
  and flight-control laws are not represented.

## Interpretation boundary

VSPAERO is used here as an independent geometry-based prior for lift, moment,
and stability-derivative structure. Although the thin-surface flow solution is
potential-flow based, this sweep retained a finite reference Reynolds number
and the saved total CD contains nonzero ``CDo`` as well as ``CDi``. It is
therefore **not** a pure induced coefficient and is not adopted as the aircraft
drag model. Aerox continues to use a separately sourced viscous drag model,
while lift and moment corrections may later be calibrated against held-out QAR
trajectory replay.

Rate derivatives in the `.stab` output are generated by perturbing dimensional
body rates at Vinf=100 solver length-units/time. Before inserting them into the
simulator, Aerox must freeze its nondimensional-rate convention and transform
the derivatives explicitly. Static derivatives can be used first.

## Release decision

- Reproducible OpenVSP geometry: **released as a development artifact**.
- Static geometry-based derivative prior: **{'released with model-form caveats' if gate['passed'] else 'held'}**.
- Viscous drag or absolute performance prediction: **not released**.
- Physical A320 six-degree-of-freedom model: **not released**.
- QAR-identified control-surface derivatives or inertia: **not available**.
"""


def main() -> None:
    args = parse_args()
    root = _openvsp_root(args.openvsp_root)
    _bootstrap_openvsp(root)
    import openvsp as vsp  # type: ignore[import-not-found]

    if args.jobs < 1:
        raise ValueError("--jobs must be positive")
    vsp.SetVSPAEROPath(str(root))
    if not vsp.CheckForVSPAERO(vsp.GetVSPAEROPath()):
        raise RuntimeError(f"VSPAERO executable was not found below {root}")

    measurements = load_measurements(args.measurements)
    nominal_raw = _nominal_geometry(measurements)
    geometry_prior = yaml.safe_load(args.geometry_prior.read_text())
    # The generated prior wraps tail quantities in uncertainty records; the
    # nominal builder values are used for solver geometry and equality-checked.
    assert geometry_prior["geometry"]["horizontal_tail"]["area_m2"]["nominal"] == nominal_raw["horizontal_tail"]["area_m2"]
    geometry = nominal_raw

    if args.keep_workdir:
        workdir = args.keep_workdir.resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        cleanup: contextlib.AbstractContextManager[Any] = contextlib.nullcontext(workdir)
    else:
        cleanup = tempfile.TemporaryDirectory(prefix="aerox-a320-vspaero-")

    with cleanup as directory:
        workdir = Path(directory)
        results: list[dict[str, Any]] = []
        model_to_publish: Path | None = None
        for mesh_name in MESHES:
            row, model_path = _run_case(
                vsp,
                workdir / f"{mesh_name}_m050",
                mesh_name,
                0.50,
                geometry,
                measurements,
                args.jobs,
                args.verbose,
            )
            results.append(row)
            if mesh_name == "medium":
                model_to_publish = model_path
        for mach in (0.20, 0.78):
            row, _ = _run_case(
                vsp,
                workdir / f"medium_m{round(mach * 100):02d}",
                "medium",
                mach,
                geometry,
                measurements,
                args.jobs,
                args.verbose,
            )
            results.append(row)

        indexed = {(row["mesh"], row["mach"]): row for row in results}
        medium = indexed[("medium", 0.50)]
        fine = indexed[("fine", 0.50)]
        coarse = indexed[("coarse", 0.50)]
        convergence = [
            {
                "derivative": name,
                "coarse": coarse[name],
                "medium": medium[name],
                "fine": fine[name],
                "coarse_medium_relative_difference": _relative_difference(coarse[name], medium[name]),
                "medium_fine_relative_difference": _relative_difference(medium[name], fine[name]),
            }
            for name in CONVERGENCE_COLUMNS
        ]
        maximum_difference = max(row["medium_fine_relative_difference"] for row in convergence)
        passed = maximum_difference <= 0.05

        result_fields = [
            "mesh",
            "mach",
            "alpha_deg",
            "vortex_loop_equations",
            "geometry_nodes",
            "CL",
            "CD_solver_total",
            "CS",
            "CMl",
            "CMm",
            "CMn",
            "static_margin",
            *DERIVATIVE_COLUMNS,
        ]
        _write_csv(args.results, sorted(results, key=lambda row: (row["mach"], row["mesh"])), result_fields)
        _write_csv(
            args.convergence,
            convergence,
            [
                "derivative",
                "coarse",
                "medium",
                "fine",
                "coarse_medium_relative_difference",
                "medium_fine_relative_difference",
            ],
        )
        if model_to_publish is None:
            raise AssertionError("medium OpenVSP model was not generated")
        args.model_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(model_to_publish, args.model_output)

        openvsp_version = vsp.GetVSPVersion().replace("OpenVSP ", "")
        vspaero_version = "7.2.2"
        version_process = subprocess.run(
            [str(root / "vspaero")],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        version_text = version_process.stdout + version_process.stderr
        match = re.search(r"VSPAERO v\.([0-9.]+)", version_text)
        if match:
            vspaero_version = match.group(1)
        prior = {
            "schema_version": 1,
            "updated": UPDATED,
            "aircraft": "Airbus A320-214",
            "configuration": "A320-200 wing-tip-fence, 2011 QAR-compatible geometry",
            "status": "geometry-based-static-prior-released-rate-prior-awaits-convention",
            "software": {
                "openvsp_version": openvsp_version,
                "vspaero_version": vspaero_version,
                "analysis_method": "thin-surface vortex lattice",
                "jobs": args.jobs,
                "wake_iterations": 5,
            },
            "source_geometry": {
                "prior": str(args.geometry_prior),
                "measurements": str(args.measurements),
                "airfoil_assumption": "symmetric four-series mean surfaces; actual A320 airfoil and twist unavailable",
                "excluded": ["fuselage", "nacelles", "pylons", "viscous drag", "aeroelasticity", "high-lift devices"],
            },
            "reference_condition": {
                "alpha_deg": ALPHA_DEG,
                "mach": list(MACH_CASES),
                "sref_m2": SREF_M2,
                "bref_m": WING_SPAN_M,
                "cref_m": WING_MAC_M,
                "planning_cg_fraction_mac": CG_FRACTION_MAC,
                "solver_vinf": 100.0,
            },
            "mesh_convergence_gate": {
                "threshold_relative": 0.05,
                "derivatives": list(CONVERGENCE_COLUMNS),
                "maximum_medium_fine_relative_difference": maximum_difference,
                "passed": passed,
                "table": str(args.convergence),
            },
            "nominal_medium_mesh": {key: medium[key] for key in result_fields if key not in {"mesh"}},
            "artifacts": {
                "model": str(args.model_output),
                "results": str(args.results),
                "report": str(args.report),
                "figure_png": str(args.figure_prefix.with_suffix(".png")),
                "figure_pdf": str(args.figure_prefix.with_suffix(".pdf")),
                "runner": "tools/build_a320_vspaero_prior.py",
            },
            "release_decision": {
                "openvsp_geometry": True,
                "static_geometry_derivative_prior": passed,
                "dimensional_rate_derivative_prior": False,
                "viscous_drag": False,
                "physical_a320_six_dof_model": False,
                "reason": "mesh convergence establishes numerical repeatability, not fidelity of omitted/proprietary geometry or identification against flight data",
            },
            "limitations": [
                "The Airbus airport-planning CAD is not an aerodynamic loft.",
                "Thin-surface VLM does not predict profile, wave, interference, or trim drag.",
                "The model omits actual airfoil camber, twist, fuselage, nacelles, pylons, aeroelasticity, and control laws.",
                "The planning CG is not the CG of either recorded flight.",
                "Rate derivatives require an explicit simulator nondimensionalization before use.",
            ],
        }
        args.parameters.parent.mkdir(parents=True, exist_ok=True)
        args.parameters.write_text(yaml.safe_dump(prior, sort_keys=False))
        args.report.write_text(_render_report(prior))
        _plot(results, convergence, args.figure_prefix)

    print(f"OpenVSP {openvsp_version}; VSPAERO {vspaero_version}")
    print(f"mesh convergence: {'PASS' if passed else 'HOLD'} ({100 * maximum_difference:.2f}% max medium-fine difference)")
    print(f"wrote {args.parameters}, {args.results}, {args.model_output}")
if __name__ == "__main__":
    main()
