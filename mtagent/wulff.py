"""Wulff-construction nanoparticle builder for cubic crystals (e.g. magnetite spinel).

ASE's built-in wulff_construction is monatomic; this does a proper *multi-element* Wulff
carve. For a cubic crystal the equilibrium shape is the intersection of facet half-spaces
{ r : n_hkl . r <= s * gamma_hkl } over all symmetry-equivalent {hkl}, where the plane
distance is proportional to the surface energy gamma. We build the spinel cell from its
space group, replicate, and carve. Lower-gamma facets ({111} for magnetite) sit closer to
the centre and dominate the shape — exactly the Wulff theorem.

Surface energies are a small literature-derived table (Materials Project does NOT provide
oxide surface energies programmatically), and are easily swapped.
"""
from __future__ import annotations

from itertools import permutations, product

import numpy as np
from ase import Atoms
from ase.spacegroup import crystal

# Magnetite Fe3O4, inverse spinel Fd-3m (227). Fe(tet) 8a, Fe(oct) 16d, O 32e (u).
MAGNETITE = {"a": 8.3941, "u": 0.2549}

# Representative DFT surface energies (J/m^2), ordered {111} < {100} < {110}.
# These are literature values (magnetite {111} is the most stable facet); swap freely.
MAGNETITE_GAMMA = {(1, 1, 1): 1.0, (1, 0, 0): 1.4, (1, 1, 0): 1.6}


def build_spinel(a: float = MAGNETITE["a"], u: float = MAGNETITE["u"],
                 tet: str = "Fe", oct: str = "Fe", anion: str = "O") -> Atoms:
    """Conventional inverse-spinel cell (56 atoms) via the space group."""
    return crystal([tet, oct, anion],
                   basis=[(0.125, 0.125, 0.125), (0.5, 0.5, 0.5), (u, u, u)],
                   spacegroup=227, cellpar=[a, a, a, 90, 90, 90], setting=2)


def _cubic_equivalents(hkl: tuple[int, int, int]) -> list[np.ndarray]:
    """All symmetry-equivalent plane normals of {hkl} under cubic m-3m."""
    out = set()
    for p in set(permutations(hkl)):
        for s in product((1, -1), repeat=3):
            v = tuple(si * pi for si, pi in zip(s, p))
            if any(v):
                out.add(v)
    return [np.array(v, float) for v in out]


def wulff_carve(cell: Atoms, gamma: dict, diameter: float) -> Atoms:
    """Carve a Wulff-shaped nanoparticle (target max diameter, Angstrom) from a cubic cell.

    `gamma` maps facet Miller indices to RELATIVE surface energies.
    In a Wulff construction each facet sits at a distance PROPORTIONAL to its
    gamma: a LOWER (positive) gamma brings that facet closer to the center and
    makes it LARGER/more prominent. A gamma <= 0 (or None) means the user wants
    that facet family REMOVED from the shape entirely; it is dropped from the
    construction (same convention as the registry template and chat edits).
    """
    if not gamma:
        raise ValueError("gamma is empty — at least one facet family is needed "
                         "to bound the Wulff shape")
    # gamma <= 0 / None reads as "no such facet" (0 surface energy would carve
    # the particle to nothing, never what the user means) — drop the family
    gamma = {hkl: float(g) for hkl, g in gamma.items()
             if g is not None and float(g) > 0}
    # cubic symmetry folds e.g. (0,0,1) and (1,0,0) into the SAME facet family —
    # two entries for one family would silently compete (tightest plane wins)
    families: dict = {}
    for hkl, g in gamma.items():
        canon = tuple(sorted((abs(int(v)) for v in hkl), reverse=True))
        if canon in families and abs(families[canon][1] - float(g)) > 1e-9:
            raise ValueError(
                f"{hkl} and {families[canon][0]} are the same cubic facet "
                f"family {{{''.join(map(str, canon))}}} but got two different "
                f"surface energies ({g} vs {families[canon][1]}) — give the "
                "family one value.")
        families.setdefault(canon, (hkl, float(g)))
    if len(families) < 2:
        kept = ", ".join("{" + "".join(map(str, c)) + "}" for c in families) \
            or "none"
        raise ValueError(
            f"a Wulff shape requires at least 2 facet families with a "
            f"positive gamma, got {kept} (gamma 0 or none removes a family). "
            "Keep a second family, or use shape sphere or cube instead.")
    a = cell.cell.lengths()[0]
    nrep = int(np.ceil(diameter / a)) + 3
    sc = cell.repeat((nrep, nrep, nrep))
    sc.set_positions(sc.get_positions() - sc.get_positions().mean(axis=0))
    pos = sc.get_positions()

    planes = [(n / np.linalg.norm(n), g)
              for hkl, g in gamma.items() for n in _cubic_equivalents(hkl)]

    def keep_mask(s: float) -> np.ndarray:
        m = np.ones(len(pos), bool)
        for n, g in planes:
            m &= (pos @ n <= s * g + 1e-6)
        return m

    # binary-search the global scale so the carved particle hits the target diameter
    lo, hi = 0.01, diameter * 2.0
    for _ in range(40):
        s = 0.5 * (lo + hi)
        sub = pos[keep_mask(s)]
        ext = (np.linalg.norm(sub, axis=1).max() * 2.0) if len(sub) else 0.0
        if ext < diameter:
            lo = s
        else:
            hi = s

    mask = keep_mask(0.5 * (lo + hi))
    np_atoms = sc[mask]
    np_atoms.set_positions(np_atoms.get_positions() - np_atoms.get_positions().mean(axis=0))
    np_atoms.set_cell(None)
    np_atoms.set_pbc(False)

    symbols = np_atoms.get_chemical_symbols()
    counts = {el: symbols.count(el) for el in set(symbols)}
    np_atoms.info["provenance"] = {
        "source": "ASE",
        "type": "wulff_spinel_nanoparticle",
        "diameter": diameter,
        "gamma": {f"{h}{k}{l}": g for (h, k, l), g in gamma.items()},
        "facets": sorted({f"{h}{k}{l}" for h, k, l in gamma}),
        "composition": counts,
        "n_atoms": len(np_atoms),
    }
    return np_atoms


def build_magnetite_wulff(diameter: float = 40.0, gamma: dict | None = None) -> Atoms:
    """Build a Wulff-faceted magnetite (Fe3O4) nanoparticle of the given diameter (A)."""
    return wulff_carve(build_spinel(), gamma or MAGNETITE_GAMMA, diameter)
