"""NP-cluster assembly via Moltemplate (deterministic lattice — NOT packmol).

Design rule: ASE builds the UNIT, Moltemplate assembles the STRUCTURE. The unit
(e.g. one nanoparticle) is defined ONCE as a reusable .lt object, then instantiated
N times with `new NP.move(x,y,z)` at deterministic simple-cubic lattice positions
(spacing = unit diameter + gap). The emitted .lt goes through the same gates as the
shell pipeline: Gate 1 (KG validation) before moltemplate.sh, Gate 2 = moltemplate.sh
runs clean, then the LAMMPS data file converts back to geometry.

packmol is only for filling solvent AROUND a pre-built cluster, never for placing NPs.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from ase import Atoms

from .kg import MoltemplateKG
from .moltemplate_emit import core_lt
from .runner import run_moltemplate, parse_lammps_data

_KG: MoltemplateKG | None = None


def _kg() -> MoltemplateKG:
    global _KG
    if _KG is None:
        _KG = MoltemplateKG()
    return _KG


def unit_diameter(unit: Atoms) -> float:
    """Bounding extent of the unit (max side of its axis-aligned bounding box)."""
    p = unit.get_positions()
    return float((p.max(axis=0) - p.min(axis=0)).max())


def lattice_positions(n: int, spacing: float, lattice: str = "sc") -> np.ndarray:
    """N deterministic superlattice sites, most-compact first.

    `spacing` is the nearest-neighbor CENTER distance (unit diameter + gap).
    `lattice`: "sc" (simple cubic), "fcc" or "bcc" — the packing of the
    nanoparticle supercrystal. Sites are sorted by distance from the lattice
    center (stable tie-break on index), so any n gives the same compact,
    reproducible arrangement — e.g. sc n=8 is exactly a 2x2x2 cube, fcc n=4
    one conventional cell. The selected sites are re-centered on the origin.
    """
    lat = (lattice or "sc").lower()
    if lat in ("sc", "simple cubic", "cubic"):
        basis, a = [(0, 0, 0)], spacing
    elif lat == "fcc":
        basis = [(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)]
        a = spacing * math.sqrt(2.0)             # fcc nn distance = a/sqrt(2)
    elif lat == "bcc":
        basis = [(0, 0, 0), (0.5, 0.5, 0.5)]
        a = spacing * 2.0 / math.sqrt(3.0)       # bcc nn distance = a*sqrt(3)/2
    else:
        raise ValueError(f"unknown superlattice {lattice!r} — sc, fcc or bcc")
    dim = math.ceil((n / len(basis)) ** (1.0 / 3.0)) + 1
    pts = np.array([(np.array(b) + (i, j, k)) for i in range(dim)
                    for j in range(dim) for k in range(dim) for b in basis],
                   dtype=float)
    center = pts.mean(axis=0)
    order = np.argsort(np.linalg.norm(pts - center, axis=1), kind="stable")
    sites = pts[order[:n]] * a
    return sites - sites.mean(axis=0)


def cluster_extent(diameter: float, n: int, gap: float = 10.0) -> float:
    """Overall extent of an n-unit cluster (for sizing the solvent box around it)."""
    if n <= 1:
        return diameter
    dim = math.ceil(n ** (1.0 / 3.0))
    return (dim - 1) * (diameter + gap) + diameter


def cluster_lt(unit_name: str, sites: np.ndarray, box: np.ndarray) -> str:
    """Master file: import the unit object, instantiate one copy per lattice site."""
    L = [f'import "{unit_name.lower()}.lt"', ""]
    for i, s in enumerate(sites):
        L.append(f"{unit_name.lower()}{i} = new {unit_name}"
                 f".move({s[0]:.6f},{s[1]:.6f},{s[2]:.6f})")
    L += ["", 'write_once("Data Boundary") {',
          f"  {box[0,0]:.4f} {box[0,1]:.4f} xlo xhi",
          f"  {box[1,0]:.4f} {box[1,1]:.4f} ylo yhi",
          f"  {box[2,0]:.4f} {box[2,1]:.4f} zlo zhi", "}"]
    return "\n".join(L)


def build_cluster(unit: Atoms, n: int, gap: float = 10.0, name: str = "NP",
                  lattice: str = "sc", workdir: str | Path = "data/work/cluster",
                  margin: float = 5.0) -> Atoms:
    """Assemble n copies of `unit` into a deterministic supercrystal via Moltemplate.

    `lattice` sets the superlattice packing of the particles: "sc" (simple
    cubic, default), "fcc" or "bcc"; nearest-neighbor center distance is the
    unit diameter + `gap`. Returns the cluster as Atoms; .info records
    spacing/extent/provenance and the emitted .lt text. Raises on Gate 1 (KG)
    or Gate 2 (moltemplate.sh) failure.
    """
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)

    dia = unit_diameter(unit)
    spacing = dia + gap
    sites = lattice_positions(n, spacing, lattice)

    centered = unit.get_positions() - unit.get_positions().mean(axis=0)
    lo = sites.min(axis=0) - dia / 2.0 - margin
    hi = sites.max(axis=0) + dia / 2.0 + margin
    box = np.column_stack([lo, hi])

    unit_file = work / f"{name.lower()}.lt"
    system_file = work / "system.lt"
    unit_file.write_text(core_lt(name, centered, unit.get_chemical_symbols()))
    system_file.write_text(cluster_lt(name, sites, box))

    # GATE 1: static KG validation of the generated .lt before running moltemplate.
    for f in (system_file, unit_file):
        rep = _kg().validate_lt(f.read_text())
        if not rep.passed:
            raise RuntimeError(f"GATE 1 FAILED in {f.name}:\n{rep.summary()}")

    # GATE 2: moltemplate.sh runs clean and produces a data file.
    data_file = run_moltemplate(system_file, work)
    elements, positions = parse_lammps_data(data_file)

    cluster = Atoms(elements, positions=positions)
    cluster.info["cluster"] = {"n_units": n, "unit_atoms": len(unit),
                               "unit_diameter": round(dia, 2), "gap": gap,
                               "spacing": round(spacing, 2), "lattice": lattice,
                               "extent": round(cluster_extent(dia, n, gap), 2)}
    cluster.info["provenance"] = {"source": "moltemplate", "type": "np_cluster",
                                  "n_units": n, "spacing": round(spacing, 2),
                                  "lattice": lattice,
                                  "system_lt": str(system_file),
                                  "lammps_data": str(data_file)}
    cluster.info["lt_files"] = {f.name: f.read_text() for f in (system_file, unit_file)}
    return cluster
