"""Clash-free packing of molecules around a core, via packmol.

Anti-hallucination axis #2 (cont.): packmol decides *where* copies go without
overlaps. This is the "shell" recipe; sibling recipes (bilayer, adsorb-on-slab)
plug in here later without touching anything downstream.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from .nanoparticle import core_radius


def _write_xyz(atoms: Atoms, path: Path) -> None:
    write(str(path), atoms, format="xyz")


def find_packmol() -> str | None:
    """Locate packmol on PATH, falling back to the running interpreter's env bin
    (the app may be launched via the env's python binary without activating it)."""
    exe = shutil.which("packmol")
    if exe:
        return exe
    candidate = Path(sys.executable).parent / "packmol"
    return str(candidate) if candidate.is_file() else None


def pack_shell(core: Atoms, molecule: Atoms, n: int,
               shell_thickness: float = 8.0, gap: float = 1.8,
               tolerance: float = 2.0, workdir: str | Path = "data/work") -> tuple[Atoms, dict]:
    """Place ``n`` copies of ``molecule`` in a spherical shell around a fixed ``core``.

    Returns (packed_atoms, meta). meta records the geometry used so verifiers and
    provenance can reason about the assembly.
    """
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH (conda install -c conda-forge packmol)")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    core_xyz, mol_xyz, out_xyz = work / "core.xyz", work / "mol.xyz", work / "packed.xyz"
    _write_xyz(core, core_xyz)
    _write_xyz(molecule, mol_xyz)

    r_core = core_radius(core)
    inner = r_core + gap                 # keep molecules off the core surface
    outer = r_core + shell_thickness     # outer bound of the shell

    inp = work / "pack.inp"
    inp.write_text(
        f"tolerance {tolerance}\n"
        f"filetype xyz\n"
        f"output {out_xyz.name}\n\n"
        f"structure {core_xyz.name}\n"
        f"  number 1\n"
        f"  fixed 0. 0. 0. 0. 0. 0.\n"
        f"end structure\n\n"
        f"structure {mol_xyz.name}\n"
        f"  number {n}\n"
        f"  inside sphere 0. 0. 0. {outer:.3f}\n"
        f"  outside sphere 0. 0. 0. {inner:.3f}\n"
        f"end structure\n"
    )

    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=300)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol failed.\nSTDOUT tail:\n{proc.stdout[-1500:]}")

    packed = read(str(out_xyz))
    meta = {
        "n_molecules": n,
        "core_atoms": len(core),
        "mol_atoms": len(molecule),
        "expected_atoms": len(core) + n * len(molecule),
        "core_radius": r_core,
        "shell_inner": inner,
        "shell_outer": outer,
        "output": str(out_xyz),
        "packmol_inp": inp.read_text(),
    }
    return packed, meta
