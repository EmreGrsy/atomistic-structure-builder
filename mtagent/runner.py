"""Run moltemplate.sh (Gate 2) and convert its LAMMPS data file back to geometry.

Gate 2 = "moltemplate.sh produced a valid data file". We do NOT run LAMMPS
(geometry-only), so the LAMMPS data file is converted straight to .xyz for viewing
and download. The data file itself is kept as an optional download.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase.data import atomic_masses, chemical_symbols


def _moltemplate_env() -> dict | None:
    """PATH with the running interpreter's env bin prepended, so moltemplate.sh
    and its helper scripts (ttree etc.) resolve even when the env isn't activated."""
    env_bin = str(Path(sys.executable).parent)
    if shutil.which("moltemplate.sh"):
        return None
    if not (Path(env_bin) / "moltemplate.sh").is_file():
        return None
    env = os.environ.copy()
    env["PATH"] = env_bin + os.pathsep + env.get("PATH", "")
    return env


def run_moltemplate(system_lt: str | Path, workdir: str | Path) -> Path:
    """Run moltemplate.sh on system.lt; return the produced .data file (Gate 2)."""
    env = _moltemplate_env()
    if env is None and shutil.which("moltemplate.sh") is None:
        raise RuntimeError("moltemplate.sh not found on PATH (pip install moltemplate)")
    system_lt = Path(system_lt)
    work = Path(workdir)
    # -nocheck: geometry-only, so skip the force-field coefficient checks
    #           (we intentionally emit no pair/bond coeffs).
    proc = subprocess.run(["moltemplate.sh", "-nocheck", system_lt.name], cwd=work,
                          capture_output=True, text=True, timeout=300, env=env)
    data = work / f"{system_lt.stem}.data"
    if proc.returncode != 0 or not data.exists():
        raise RuntimeError("GATE 2 FAILED (moltemplate.sh):\n"
                           f"{proc.stderr[-1500:] or proc.stdout[-1500:]}")
    return data


def _mass_to_element(mass: float) -> str:
    diffs = np.abs(atomic_masses[1:] - mass)
    return chemical_symbols[int(np.argmin(diffs)) + 1]


def parse_lammps_data(data_path: str | Path) -> tuple[list[str], np.ndarray]:
    """Parse a LAMMPS 'full' data file -> (elements, positions), sorted by atom id.

    Elements are inferred from the Masses section (topology, no force field needed).
    """
    text = Path(data_path).read_text()
    keywords = {"Masses", "Atoms", "Velocities", "Bonds", "Angles",
                "Dihedrals", "Impropers", "Pair Coeffs", "Bond Coeffs"}
    masses: dict[int, float] = {}
    rows: list[tuple[int, int, float, float, float]] = []  # id, type, x, y, z
    section = None
    for raw in text.splitlines():
        body = raw.split("#")[0].rstrip()
        key = body.strip()
        if key in keywords:
            section = key
            continue
        if not key:
            continue
        tok = key.split()
        if section == "Masses":
            masses[int(tok[0])] = float(tok[1])
        elif section == "Atoms":                        # full: id mol type q x y z
            rows.append((int(tok[0]), int(tok[2]),
                         float(tok[4]), float(tok[5]), float(tok[6])))

    rows.sort(key=lambda r: r[0])
    elem_of_type = {t: _mass_to_element(m) for t, m in masses.items()}
    elements = [elem_of_type[r[1]] for r in rows]
    positions = np.array([[r[2], r[3], r[4]] for r in rows], dtype=float)
    return elements, positions


def lammps_data_to_xyz(data_path: str | Path, out_xyz: str | Path) -> tuple[list[str], np.ndarray]:
    """Convert a LAMMPS data file to .xyz. Returns (elements, positions)."""
    elements, positions = parse_lammps_data(data_path)
    lines = [str(len(elements)), "converted from moltemplate LAMMPS data"]
    for el, p in zip(elements, positions):
        lines.append(f"{el} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    Path(out_xyz).write_text("\n".join(lines) + "\n")
    return elements, positions
