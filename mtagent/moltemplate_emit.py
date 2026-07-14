"""Emit Moltemplate (.lt) files from a packed geometry.

This is what makes the project a *Moltemplate* agent rather than "packmol + xyz":
the shell molecule is defined ONCE as a reusable `.lt` object, then instantiated
N times with per-copy `.rot(...).move(...)` transforms. Each transform is recovered
from the packmol coordinates with the Kabsch algorithm (best-fit rigid alignment of
the reference molecule onto each packed copy).

Geometry-only: we emit Data Atoms / Data Bonds / Data Masses / Data Boundary, but no
force-field sections (no pair/bond coeffs). Masses are included so the LAMMPS data
file is well-formed (they are topology, not a force field).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.data import atomic_masses, chemical_symbols


# ---- rigid-transform recovery -------------------------------------------------

def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rotation matrix R (proper, det=+1) best-mapping centered P onto centered Q."""
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    return Vt.T @ D @ U.T


def rotation_to_axis_angle(R: np.ndarray) -> tuple[float, np.ndarray]:
    """Convert a rotation matrix to (angle_degrees, unit_axis) for Moltemplate .rot()."""
    angle = np.arccos(np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0))
    if angle < 1e-6:
        return 0.0, np.array([0.0, 0.0, 1.0])
    if abs(angle - np.pi) < 1e-4:                      # 180 deg: axis from (R+I)/2
        M = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(M), 0.0, None))
        if axis[0] > 1e-6:
            axis[1] = np.copysign(axis[1], M[0, 1])
            axis[2] = np.copysign(axis[2], M[0, 2])
        elif axis[1] > 1e-6:
            axis[2] = np.copysign(axis[2], M[1, 2])
        return 180.0, axis / np.linalg.norm(axis)
    axis = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return float(np.degrees(angle)), axis / (2.0 * np.sin(angle))


# ---- .lt text generation ------------------------------------------------------

def _masses_block(elements: list[str], indent: str = "  ") -> list[str]:
    out = [f'{indent}write_once("Data Masses") {{']
    for el in sorted(set(elements)):
        m = atomic_masses[chemical_symbols.index(el)]
        out.append(f"{indent}  @atom:{el} {m:.4f}")
    out.append(f"{indent}}}")
    return out


def molecule_lt(name: str, positions: np.ndarray, elements: list[str],
                bonds: list[tuple[int, int, int]]) -> str:
    """A reusable molecule object, defined at its centered reference geometry."""
    L = [f"{name} {{"]
    L += _masses_block(elements)
    L.append('  write("Data Atoms") {')
    for i, (el, p) in enumerate(zip(elements, positions)):
        L.append(f"    $atom:a{i} $mol:. @atom:{el} 0.0 "
                 f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    L.append("  }")
    if bonds:
        L.append('  write("Data Bonds") {')
        for j, (a, b, _order) in enumerate(bonds):
            btype = "".join(sorted((elements[a], elements[b])))
            L.append(f"    $bond:b{j} @bond:{btype} $atom:a{a} $atom:a{b}")
        L.append("  }")
    L.append("}")
    return "\n".join(L)


def core_lt(name: str, positions: np.ndarray, elements: list[str]) -> str:
    """A single rigid core block (e.g. the nanoparticle), no intramolecular bonds."""
    L = [f"{name} {{"]
    L += _masses_block(elements)
    L.append('  write("Data Atoms") {')
    for i, (el, p) in enumerate(zip(elements, positions)):
        L.append(f"    $atom:a{i} $mol:. @atom:{el} 0.0 "
                 f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}")
    L.append("  }")
    L.append("}")
    return "\n".join(L)


def system_lt(core_name: str, mol_name: str,
              placements: list[tuple[float, np.ndarray, np.ndarray]],
              box: np.ndarray) -> str:
    """Master file: import objects, place the core + N transformed molecules, set box."""
    L = [f'import "{core_name.lower()}.lt"',
         f'import "{mol_name.lower()}.lt"', "",
         f"core = new {core_name}"]
    for i, (ang, axis, t) in enumerate(placements):
        L.append(f"mol{i} = new {mol_name}"
                 f".rot({ang:.4f},{axis[0]:.6f},{axis[1]:.6f},{axis[2]:.6f})"
                 f".move({t[0]:.6f},{t[1]:.6f},{t[2]:.6f})")
    L += ["", 'write_once("Data Boundary") {',
          f"  {box[0,0]:.4f} {box[0,1]:.4f} xlo xhi",
          f"  {box[1,0]:.4f} {box[1,1]:.4f} ylo yhi",
          f"  {box[2,0]:.4f} {box[2,1]:.4f} zlo zhi", "}"]
    return "\n".join(L)


# ---- top-level: packed geometry -> .lt files ---------------------------------

def emit_assembly(core: Atoms, molecule: Atoms, packed: Atoms, n_molecules: int,
                  workdir, core_name: str = "Core", mol_name: str = "Mol",
                  margin: float = 5.0) -> dict:
    """Write core.lt, mol.lt, system.lt reproducing the packed geometry. Returns paths."""
    from pathlib import Path
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    ref = molecule.get_positions() - molecule.get_positions().mean(axis=0)
    mol_elems = molecule.get_chemical_symbols()
    bonds = molecule.info.get("bonds", [])
    n_core, n_mol = len(core), len(molecule)

    # recover per-copy transforms from the packed shell atoms
    shell = packed.get_positions()[n_core:]
    placements = []
    for k in range(n_molecules):
        T = shell[k * n_mol:(k + 1) * n_mol]
        R = kabsch(ref, T)
        ang, axis = rotation_to_axis_angle(R)
        placements.append((ang, axis, T.mean(axis=0)))

    p = packed.get_positions()
    box = np.column_stack([p.min(axis=0) - margin, p.max(axis=0) + margin])

    (work / f"{core_name.lower()}.lt").write_text(
        core_lt(core_name, core.get_positions(), core.get_chemical_symbols()))
    (work / f"{mol_name.lower()}.lt").write_text(
        molecule_lt(mol_name, ref, mol_elems, bonds))
    (work / "system.lt").write_text(
        system_lt(core_name, mol_name, placements, box))

    return {"system_lt": str(work / "system.lt"),
            "core_lt": str(work / f"{core_name.lower()}.lt"),
            "mol_lt": str(work / f"{mol_name.lower()}.lt"),
            "n_placements": len(placements)}
