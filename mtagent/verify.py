"""Gate 3 verifiers — validate a built geometry the same way every time.

Deterministic checks that decide whether an executed snippet's structure is
acceptable: it has atoms, all coordinates are finite, and no two atoms overlap
non-physically. Runs on every constituent build and on the showcase assembly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class Check:
    name: str
    passed: bool
    detail: str


@dataclass
class Report:
    passed: bool
    checks: list[Check]
    min_distance: float

    def summary(self) -> str:
        return "  ".join(f"{'✓' if c.passed else '✗'}{c.name}" for c in self.checks)


def min_interatomic_distance(positions: np.ndarray) -> float:
    """Smallest distance between any two atoms (clash detector)."""
    if len(positions) < 2:
        return float("inf")
    d, _ = cKDTree(positions).query(positions, k=2)
    return float(d[:, 1].min())


def verify_atoms(atoms, clash_floor: float = 0.6) -> Report:
    """Gate 3 for any built structure: non-empty, finite coordinates, no clashes."""
    positions = atoms.get_positions()
    checks = [Check("has_atoms", len(positions) > 0, f"{len(positions)} atoms")]

    finite = len(positions) > 0 and bool(np.isfinite(positions).all())
    checks.append(Check("finite_coords", finite,
                        "all finite" if finite else "NaN/inf present"))

    mind = min_interatomic_distance(positions) if finite else 0.0
    checks.append(Check("no_clash", mind >= clash_floor,
                        f"min nn dist {mind:.3f} Å (floor {clash_floor})"))

    return Report(passed=all(c.passed for c in checks), checks=checks, min_distance=mind)
