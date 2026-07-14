"""Generic coordination analysis + surface hydroxylation (element-agnostic).

Ports the method in the user's hydroxylate.tcl to ASE, generalized to ANY oxide/ionic
nanoparticle (not hardcoded to magnetite):

  1. Coordination number of each cation = anions within `cutoff`. Default is radii-based
     (per-element covalent radii, cutoff=None) which robustly isolates the first shell for
     any composition; a fixed cutoff (e.g. 3.0 A, as in the original TCL) can be passed but
     is structure-specific (3.0 A grabs 2nd-shell O in an ideal crystal).
  2. Bulk coordination is INFERRED from the structure (the common high-coordination values),
     so surface undercoordination is measured relative to the material itself.
  3. Undercoordinated sites = cations with coordination <= `threshold` (the "vastly
     undercoordinated" 0/1/2-coordinate ions), taken worst-first.
  4. Each selected site is capped with -OH placed RADIALLY OUTWARD from the NP centre of
     mass (as in the TCL): O at cation_radius + oh_distance, H one bond further out.

Geometry-only: adds O/H atoms + O-H bonds; charges are not handled here.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from ase import Atom, Atoms
from ase.neighborlist import NeighborList, natural_cutoffs


def coordination_numbers(atoms: Atoms, cation="Fe", anion="O",
                         cutoff: float | None = None) -> dict[int, int]:
    """Per-cation coordination number (count of anion neighbours within cutoff)."""
    if cutoff is None:
        cutoffs = natural_cutoffs(atoms, mult=1.2)
    else:
        cutoffs = [cutoff / 2.0] * len(atoms)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
    nl.update(atoms)

    syms = atoms.get_chemical_symbols()
    cations = {cation} if isinstance(cation, str) else set(cation)
    anions = {anion} if isinstance(anion, str) else set(anion)

    coord: dict[int, int] = {}
    for i in range(len(atoms)):
        if syms[i] in cations:
            idx, _ = nl.get_neighbors(i)
            coord[i] = sum(syms[j] in anions for j in idx)
    return coord


def infer_bulk_coordination(coord: dict[int, int]) -> list[int]:
    """Infer the bulk (full) coordination value(s): the frequent high-coordination modes."""
    dist = Counter(coord.values())
    if not dist:
        return []
    peak = max(dist.values())
    # bulk = coordination values that are common (>=25% of the peak) and not tiny
    return sorted(v for v, n in dist.items() if n >= 0.25 * peak and v >= 3)


def select_undercoordinated(coord: dict[int, int], threshold: int = 2) -> list[int]:
    """Cation indices with coordination <= threshold, worst (lowest) first."""
    under = [(i, c) for i, c in coord.items() if c <= threshold]
    return [i for i, _ in sorted(under, key=lambda x: x[1])]


def hydroxylate(atoms: Atoms, cation="Fe", anion="O", cutoff: float | None = None,
                threshold: int = 2, oh_distance: float = 2.5, oh_bond: float = 0.96,
                max_sites: int | None = None) -> Atoms:
    """Cap undercoordinated surface cations with radial -OH groups. Returns a new Atoms.

    The result carries .info['hydroxylation'] with the sites, O-H bonds, and coordination
    stats; O and H atoms are appended as consecutive (O, H) pairs.
    """
    out = atoms.copy()
    pos = out.get_positions()
    com = pos.mean(axis=0)

    coord = coordination_numbers(out, cation, anion, cutoff)
    bulk = infer_bulk_coordination(coord)
    sites = select_undercoordinated(coord, threshold)
    if max_sites is not None:
        sites = sites[:max_sites]

    n0 = len(out)
    bonds: list[tuple[int, int]] = []
    add = Atoms()
    for k, i in enumerate(sites):
        r = pos[i] - com
        d = float(np.linalg.norm(r))
        direction = r / d if d > 1e-6 else np.array([0.0, 0.0, 1.0])
        o_pos = com + (d + oh_distance) * direction
        h_pos = o_pos + oh_bond * direction
        add += Atom("O", o_pos)
        add += Atom("H", h_pos)
        bonds.append((n0 + 2 * k, n0 + 2 * k + 1))     # O-H bond
    out += add

    out.info["hydroxylation"] = {
        "cation": cation, "anion": anion, "cutoff": cutoff, "threshold": threshold,
        "bulk_coordination": bulk,
        "coord_distribution": dict(sorted(Counter(coord.values()).items())),
        "n_cations": len(coord),
        "n_undercoordinated": len(sites),
        "oh_bonds": bonds,
        "oh_distance": oh_distance,
    }
    return out
