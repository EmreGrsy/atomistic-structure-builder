"""Build metal nanoparticle cores with ASE (bounded tool API, no LLM).

Anti-hallucination axis #2: geometry from tested library calls, not generated code.
A spherical FCC nanoparticle is carved from a bulk crystal so the diameter is
predictable and the result is centered on the origin (ready for shell packing).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk

# Experimental FCC lattice constants (Angstrom).
# Fe here is idealized gamma-iron (FCC), used as a stand-in for an iron-oxide core;
# real magnetite is a spinel, not FCC.
_LATTICE = {"Au": 4.078, "Ag": 4.085, "Pt": 3.924, "Pd": 3.891, "Cu": 3.615,
            "Al": 4.050, "Fe": 3.585}


def _fcc_a_from_ase(element: str) -> float | None:
    """FCC lattice constant from ASE's built-in reference-state database."""
    from ase.data import atomic_numbers, reference_states
    z = atomic_numbers.get(element)
    ref = reference_states[z] if z is not None else None
    if ref and ref.get("symmetry") == "fcc":
        return float(ref["a"])
    return None


def build_fcc_sphere(element: str = "Au", diameter: float = 20.0,
                     a: float | None = None) -> Atoms:
    """Carve a spherical FCC nanoparticle of the given diameter (Angstrom).

    Returns an ASE Atoms centered on the origin, with provenance in .info.
    """
    if a is None:
        a = _LATTICE.get(element) or _fcc_a_from_ase(element)
        if a is None:
            raise ValueError(
                f"No FCC lattice constant on record for {element!r} (ASE reference "
                f"state is not FCC) — pass a=..., or use "
                "mtagent.nanostructures.build_element_sphere for the ground-state "
                "crystal structure.")

    n_rep = int(np.ceil(diameter / a)) + 2
    crystal = bulk(element, "fcc", a=a, cubic=True).repeat((n_rep, n_rep, n_rep))

    pos = crystal.get_positions()
    crystal.set_positions(pos - pos.mean(axis=0))          # center on origin
    radius = diameter / 2.0
    sphere = crystal[np.linalg.norm(crystal.get_positions(), axis=1) <= radius]

    sphere.info["provenance"] = {
        "source": "ASE",
        "type": "fcc_nanoparticle",
        "element": element,
        "diameter": diameter,
        "lattice_a": a,
        "n_atoms": len(sphere),
    }
    from .nanostructures import _no_empty
    return _no_empty(sphere, "spherical carve", diameter)


def core_radius(atoms: Atoms) -> float:
    """Max distance of any atom from the origin (assumes centered structure)."""
    return float(np.linalg.norm(atoms.get_positions(), axis=1).max())


def build_fcc_sphere_with_fault(element: str = "Fe", diameter: float = 40.0,
                                a: float | None = None,
                                fault_fraction: float = 0.5) -> Atoms:
    """Carve a spherical FCC nanoparticle containing one stacking fault.

    FCC is ABCABC… close-packing of (111) planes. We stack (111) layers explicitly;
    a fault is introduced by advancing the stacking registry by one extra step at a
    chosen plane (ABC…AB|CAB… — a Shockley-partial-type intrinsic fault). Layer
    offsets stay in {0, s, 2s} so the layers remain centered and clash-free
    (nearest-neighbour distance is a/sqrt(2) everywhere, including across the fault).

    ``fault_fraction`` in (0,1) picks the fault plane along the stacking axis.
    Returns an ASE Atoms centered on the origin; the stacking string and fault plane
    are recorded in .info['provenance'].
    """
    if a is None:
        a = _LATTICE.get(element) or _fcc_a_from_ase(element)
        if a is None:
            raise ValueError(f"No FCC lattice constant on record for {element!r}; "
                             "pass a=...")

    d = a / np.sqrt(2.0)          # in-plane nearest-neighbour distance
    c = a / np.sqrt(3.0)          # (111) interlayer spacing
    v1 = np.array([d, 0.0, 0.0])
    v2 = np.array([d / 2.0, d * np.sqrt(3.0) / 2.0, 0.0])
    s = (v1 + v2) / 3.0           # stacking shift A->B->C

    radius = diameter / 2.0
    n_lay = int(np.ceil(diameter / c)) + 3
    n2d = int(np.ceil(diameter / d)) + 3
    fault_layer = min(max(int(round(fault_fraction * (n_lay - 1))), 1), n_lay - 1)

    positions, sequence = [], []
    ij = range(-n2d, n2d + 1)
    for L in range(n_lay):
        t = L + (1 if L >= fault_layer else 0)   # extra registry step above the fault
        k = t % 3
        off = k * s
        sequence.append("ABC"[k])
        z = L * c
        for i in ij:
            for j in ij:
                p = i * v1 + j * v2 + off
                positions.append([p[0], p[1], z])

    pos = np.array(positions, dtype=float)
    pos -= pos.mean(axis=0)
    sphere = Atoms(symbols=[element] * int((np.linalg.norm(pos, axis=1) <= radius).sum()),
                   positions=pos[np.linalg.norm(pos, axis=1) <= radius])

    sphere.info["provenance"] = {
        "source": "ASE",
        "type": "fcc_nanoparticle_stacking_fault",
        "element": element,
        "diameter": diameter,
        "lattice_a": a,
        "fault_layer": fault_layer,
        "n_layers": n_lay,
        "stacking": "".join(sequence),
        "n_atoms": len(sphere),
    }
    return sphere
