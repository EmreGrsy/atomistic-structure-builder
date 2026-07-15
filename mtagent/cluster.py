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

import itertools
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


def _lattice_basis(lattice: str, spacing: float) -> tuple[list, float]:
    """(fractional basis, conventional cell edge) for a superlattice type.
    `spacing` is the nearest-neighbor CENTER distance."""
    lat = (lattice or "sc").lower()
    if lat in ("sc", "simple cubic", "cubic"):
        return [(0, 0, 0)], spacing
    if lat == "fcc":
        return ([(0, 0, 0), (0.5, 0.5, 0), (0.5, 0, 0.5), (0, 0.5, 0.5)],
                spacing * math.sqrt(2.0))        # fcc nn distance = a/sqrt(2)
    if lat == "bcc":
        return ([(0, 0, 0), (0.5, 0.5, 0.5)],
                spacing * 2.0 / math.sqrt(3.0))  # bcc nn distance = a*sqrt(3)/2
    raise ValueError(f"unknown superlattice {lattice!r} — sc, fcc or bcc")


def stacking_positions(sequence: str, per_layer: int, spacing: float) -> np.ndarray:
    """Close-packed layer stacking with an EXPLICIT sequence, e.g. 'ABCABCABAB'
    — an fcc crystal with a stacking fault. Each layer is a compact 2D
    hexagonal patch of `per_layer` particles (nearest-neighbor distance
    `spacing`); A/B/C are the three close-packed layer registries; the
    interlayer distance is spacing*sqrt(2/3)."""
    seq = sequence.strip().upper()
    if not seq or any(ch not in "ABC" for ch in seq):
        raise ValueError(f"stacking sequence must use only A/B/C, got {sequence!r}")
    a1 = np.array([1.0, 0.0]) * spacing
    a2 = np.array([0.5, math.sqrt(3.0) / 2.0]) * spacing
    registry = {"A": np.zeros(2), "B": (a1 + a2) / 3.0, "C": 2.0 * (a1 + a2) / 3.0}
    # one compact hexagonal patch reused by every layer
    grid = np.array([i * a1 + j * a2 for i in range(-3, 4) for j in range(-3, 4)])
    order = np.argsort(np.linalg.norm(grid - grid.mean(axis=0), axis=1), kind="stable")
    patch = grid[order[:per_layer]]
    dz = spacing * math.sqrt(2.0 / 3.0)
    sites = np.array([[*(xy + registry[ch]), k * dz]
                      for k, ch in enumerate(seq) for xy in patch])
    return sites - sites.mean(axis=0)


def lattice_positions(n: int, spacing: float, lattice: str = "sc") -> np.ndarray:
    """N deterministic superlattice sites, most-compact first.

    `spacing` is the nearest-neighbor CENTER distance (unit diameter + gap).
    `lattice`: "sc" (simple cubic), "fcc" or "bcc" — the packing of the
    nanoparticle supercrystal. Sites are sorted by distance from the lattice
    center (stable tie-break on index), so any n gives the same compact,
    reproducible arrangement — e.g. sc n=8 is exactly a 2x2x2 cube, fcc n=4
    one conventional cell. The selected sites are re-centered on the origin.
    """
    basis, a = _lattice_basis(lattice, spacing)
    dim = math.ceil((n / len(basis)) ** (1.0 / 3.0)) + 2
    pts = np.array([(np.array(b) + (i, j, k)) for i in range(dim)
                    for j in range(dim) for k in range(dim) for b in basis],
                   dtype=float)
    mid = pts.mean(axis=0)
    # which n-subset is tightest depends on where the probe center sits (on a
    # lattice site vs between sites): a site-centered sort turns e.g. sc n=8
    # into a spiky octahedral star instead of the 2x2x2 cube. Try the
    # half-grid offsets and keep the selection with the smallest spread.
    best = None
    for off in itertools.product((0.0, 0.5), repeat=3):
        order = np.argsort(np.linalg.norm(pts - (mid + np.array(off)), axis=1),
                           kind="stable")
        sel = pts[order[:n]]
        spread = sel.max(axis=0) - sel.min(axis=0)
        score = (round(float(spread.max()), 6), round(float(spread.sum()), 6),
                 round(float(np.linalg.norm(
                     sel - sel.mean(axis=0), axis=1).sum()), 6))
        if best is None or score < best[0]:
            best = (score, sel)
    sites = best[1] * a
    return sites - sites.mean(axis=0)


def cluster_extent(diameter: float, n: int, gap: float = 10.0,
                   lattice: str = "sc") -> float:
    """Overall extent of an n-unit cluster (for sizing the solvent box around
    it) — from the ACTUAL lattice sites, not a cubic-grid overestimate (which
    oversized solvent boxes around close-packed supercrystals)."""
    if n <= 1:
        return diameter
    try:
        sites = lattice_positions(int(n), diameter + gap, lattice)
        return float((sites.max(axis=0) - sites.min(axis=0)).max()) + diameter
    except Exception:
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
    lat = str(lattice or "sc")
    stacking = len(lat) > 3 and all(c in "ABCabc" for c in lat)
    if stacking:                       # explicit layer sequence (e.g. ABCABAB)
        per_layer = max(1, round(n / len(lat)))
        n = per_layer * len(lat)
        sites = stacking_positions(lat, per_layer, spacing)
        lo = sites.min(axis=0) - dia / 2.0 - margin
        hi_ = sites.max(axis=0) + dia / 2.0 + margin
        box = np.column_stack([lo, hi_])
        periodic = False
    else:
        basis, a_lat = _lattice_basis(lattice, spacing)
        k = round((n / len(basis)) ** (1.0 / 3.0))
        periodic = k >= 1 and len(basis) * k ** 3 == n
        if periodic:
            # n fills complete conventional cells -> a TRUE periodic
            # supercrystal: cell edge = k*a, so the structure tiles seamlessly
            # under PBC (boundary-image neighbor distance == in-crystal spacing)
            sites = np.array([(np.array(b) + (i, j, m)) * a_lat
                              for i in range(k) for j in range(k)
                              for m in range(k) for b in basis])
            cell_len = k * a_lat
            box = np.column_stack([np.zeros(3), np.full(3, cell_len)])
        else:
            sites = lattice_positions(n, spacing, lattice)
            lo = sites.min(axis=0) - dia / 2.0 - margin
            hi = sites.max(axis=0) + dia / 2.0 + margin
            box = np.column_stack([lo, hi])

    centered = unit.get_positions() - unit.get_positions().mean(axis=0)

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
    if periodic:
        cluster.set_cell(np.eye(3) * cell_len)
        cluster.set_pbc(True)
    cluster.info["cluster"] = {"n_units": n, "unit_atoms": len(unit),
                               "unit_diameter": round(dia, 2), "gap": gap,
                               "spacing": round(spacing, 2), "lattice": lattice,
                               "periodic": periodic,
                               "extent": round(cluster_extent(dia, n, gap,
                                                              lattice), 2)}
    cluster.info["provenance"] = {"source": "moltemplate", "type": "np_cluster",
                                  "n_units": n, "spacing": round(spacing, 2),
                                  "lattice": lattice,
                                  "system_lt": str(system_file),
                                  "lammps_data": str(data_file)}
    cluster.info["lt_files"] = {f.name: f.read_text() for f in (system_file, unit_file)}
    return cluster
