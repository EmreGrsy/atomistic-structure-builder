"""Solvent box builder + generic carve-and-insert solvation (geometry-only).

- build_solvent_box: fill a periodic cube with a molecule at a target density (packmol).
- solvate: the generic combine — carve a cavity in the solvent box and insert the solute,
  deleting whole solvent molecules that clash with it. Identical operation for ANY
  solute-in-solvent interface.

Reference densities are a small table (the "reference density" a literature agent would
supply later); pass `density=` to override.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write
from scipy.spatial import cKDTree

from .packing import find_packmol

_NA = 6.02214076e23

# g/cm^3 — placeholder reference densities (literature agent supplies these later).
SOLVENT_DENSITY = {"water": 1.00, "oleic acid": 0.895, "methanol": 0.792,
                   "ethanol": 0.789, "hexane": 0.655, "toluene": 0.867}


def n_for_density(molecule: Atoms, box_size: float, density: float) -> int:
    """Number of molecules to reach `density` (g/cm^3) in a cubic box of side `box_size` (A)."""
    molar_mass = float(molecule.get_masses().sum())          # g/mol
    v_cm3 = (box_size * 1e-8) ** 3
    return max(1, round(density * v_cm3 * _NA / molar_mass))


def build_solvent_box(molecule: Atoms, box_size: float = 40.0, density: float | None = None,
                      n: int | None = None, name: str = "solvent", tolerance: float = 2.0,
                      workdir: str | Path = "data/work", max_atoms: int = 8000,
                      timeout: int = 180) -> Atoms:
    """Pack `n` (or density-derived) copies of `molecule` into a periodic cube (packmol).

    `max_atoms` caps the fill so packmol stays fast for large molecules (e.g. oleic acid);
    when capped the box is a *partial* fill (lower density) — reported in .info['capped'].
    """
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")
    if box_size < 15.0:
        raise ValueError(
            f"box_size={box_size:g} Å is smaller than a single solvation "
            "shell. Sizes are in Ångström (1 nm = 10 Å); a typical box is "
            "30 to 60 Å.")
    if density is not None and float(density) <= 0:
        density = None                  # LLM "auto" spelled as density=0
    target_density = density
    if n is not None:
        try:                            # LLMs express "auto" as 0/None/"auto"
            n = int(n)
        except (TypeError, ValueError):
            n = None
        if n is not None and n < 1:
            n = None                    # 0 or negative = fill at density
    explicit_n = n is not None
    if n is None:
        if density is None:
            density = SOLVENT_DENSITY.get(name, 1.0)
            target_density = density
        n = n_for_density(molecule, box_size, density)
    # the cap protects density AUTOFILLS from packmol blowups; a count the
    # user explicitly requested is honored as given
    capped = (not explicit_n) and n * len(molecule) > max_atoms
    if capped:
        n = max(1, max_atoms // len(molecule))

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    mol_xyz, out_xyz = work / "solv_mol.xyz", work / "solv_box.xyz"
    write(str(mol_xyz), molecule, format="xyz")
    h = box_size / 2.0
    inp = work / "solv.inp"
    inp.write_text(
        f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
        f"structure {mol_xyz.name}\n  number {n}\n"
        f"  inside box {-h:.3f} {-h:.3f} {-h:.3f} {h:.3f} {h:.3f} {h:.3f}\n"
        f"end structure\n")
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol solvent box failed:\n{proc.stdout[-1200:]}")

    box = read(str(out_xyz))
    box.set_cell([box_size, box_size, box_size])
    box.set_pbc(True)
    box.set_positions(box.get_positions() - box.get_positions().mean(axis=0))
    actual_density = n * float(molecule.get_masses().sum()) / (_NA * (box_size * 1e-8) ** 3)
    box.info["n_per_molecule"] = len(molecule)
    box.info["n_molecules"] = n
    box.info["capped"] = capped
    box.info["packmol_inp"] = inp.read_text()
    box.info["provenance"] = {"source": "packmol", "type": "solvent_box", "name": name,
                              "box_size": box_size, "density": round(actual_density, 3),
                              "target_density": target_density, "n_molecules": n,
                              "capped": capped}
    return box


def add_to_box(box: Atoms, molecule: Atoms, n: int, name: str = "solute",
               tolerance: float = 2.0, workdir: str | Path = "data/work",
               timeout: int = 900) -> Atoms:
    """Dissolve `n` copies of `molecule` INTO an existing solvent box.

    Packmol keeps the box contents fixed and packs the new species into the
    same periodic cube. The result records per species molecule sizes so
    solvate() can carve every species correctly."""
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")
    n = int(n)
    L = float(box.cell.lengths()[0])
    h = L / 2.0 - 0.5
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    box_xyz, mol_xyz, out_xyz = (work / "mix_box.xyz", work / "mix_mol.xyz",
                                 work / "mix_out.xyz")
    write(str(box_xyz), box, format="xyz")
    m = molecule.copy()
    m.set_positions(m.get_positions() - m.get_positions().mean(axis=0))
    write(str(mol_xyz), m, format="xyz")
    inp = work / "mix.inp"
    inp.write_text(
        f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
        f"structure {box_xyz.name}\n  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"structure {mol_xyz.name}\n  number {n}\n"
        f"  inside box {-h:.3f} {-h:.3f} {-h:.3f} {h:.3f} {h:.3f} {h:.3f}\n"
        "end structure\n")
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol add_to_box failed:\n{proc.stdout[-1200:]}")
    mixed = read(str(out_xyz))
    mixed.set_cell(box.get_cell())
    mixed.set_pbc(True)
    blocks = box.info.get("species_blocks") or [
        [int(box.info.get("n_per_molecule", 1)),
         int(box.info.get("n_molecules",
                          len(box) // max(1, box.info.get("n_per_molecule", 1))))]]
    mixed.info.update(box.info)
    mixed.info["species_blocks"] = blocks + [[len(molecule), n]]
    mixed.info["packmol_inp"] = inp.read_text()
    return mixed


def solvate(solute: Atoms, solvent_box: Atoms, clash: float = 2.5) -> Atoms:
    """Carve a cavity in `solvent_box` for `solute` and insert it (delete clashing solvent).

    `clash` is the solute–solvent CONTACT CUTOFF in Å (typical 2–3): solvent
    molecules closer than this to any solute atom are removed. It is NOT the
    box size — the box size is a property of `solvent_box` itself
    (build_solvent_box(box_size=...)).
    """
    if clash > 10.0:
        raise ValueError(
            f"clash={clash:g} Å is not a contact distance — it would delete "
            "all solvent near the solute. clash is the solute–solvent overlap "
            "cutoff (typical 2–3 Å); to control the amount of solvent, size "
            "the box via build_solvent_box(box_size=...) instead.")
    solute = solute.copy()
    solute.set_positions(solute.get_positions() - solute.get_positions().mean(axis=0))

    sp = solvent_box.get_positions()
    tree = cKDTree(solute.get_positions())
    dmin, _ = tree.query(sp)                                  # nearest solute atom per solvent atom
    # per species molecule sizes: a plain box is one block; a mixed box
    # (add_to_box) carries several
    blocks = solvent_box.info.get("species_blocks") or [
        [int(solvent_box.info.get("n_per_molecule", 1)),
         len(sp) // max(1, int(solvent_box.info.get("n_per_molecule", 1)))]]
    keep_atom = np.zeros(len(sp), dtype=bool)
    n_kept = n_removed = 0
    off = 0
    for k, count in blocks:
        seg = dmin[off:off + k * count].reshape(count, k).min(axis=1)
        keep = seg >= clash
        keep_atom[off:off + k * count] = np.repeat(keep, k)
        n_kept += int(keep.sum())
        n_removed += int((~keep).sum())
        off += k * count
    keep_mol = None                                           # totals below

    kept = solvent_box[keep_atom]
    combined = solute + kept
    combined.set_cell(solvent_box.get_cell())
    combined.set_pbc(True)
    combined.info["solvation"] = {
        "solute_atoms": len(solute),
        "solvent_molecules_kept": n_kept,
        "solvent_molecules_removed": n_removed,
        "total_atoms": len(combined),
        "clash": clash,
    }
    return combined
