"""Fetch molecular structures from PubChem (provenance-tracked, cached).

Anti-hallucination axis #1: coordinates come from a real database, never invented.
Every returned structure carries ``atoms.info['provenance']`` so downstream verifiers
(Gate 3) can confirm every atom traces back to a source.
"""
from __future__ import annotations

from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from ase import Atoms

_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


def _cache_dir() -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE


def name_to_cid(name: str) -> int:
    """Resolve a compound name to its PubChem CID."""
    r = requests.get(f"{_REST}/compound/name/{quote(name)}/cids/TXT", timeout=30)
    r.raise_for_status()
    return int(r.text.strip().splitlines()[0])


def fetch_sdf(cid: int) -> str:
    """Download the 3D SDF record for a CID, caching to data/cache."""
    path = _cache_dir() / f"cid_{cid}_3d.sdf"
    if path.exists():
        return path.read_text()
    r = requests.get(f"{_REST}/compound/cid/{cid}/record/SDF",
                     params={"record_type": "3d"}, timeout=60)
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


def parse_sdf(text: str) -> tuple[list[str], np.ndarray, list[tuple[int, int, int]]]:
    """Parse an MDL V2000 SDF block into (elements, positions, bonds).

    Bonds are 0-indexed (i, j, order). Deliberately dependency-free so we don't
    rely on an optional SDF reader being present.
    """
    lines = text.splitlines()
    counts = lines[3]
    n_atoms = int(counts[0:3])
    n_bonds = int(counts[3:6])

    elements: list[str] = []
    positions = []
    for ln in lines[4:4 + n_atoms]:
        tok = ln.split()
        positions.append([float(tok[0]), float(tok[1]), float(tok[2])])
        elements.append(tok[3])

    bonds: list[tuple[int, int, int]] = []
    for ln in lines[4 + n_atoms:4 + n_atoms + n_bonds]:
        tok = ln.split()
        bonds.append((int(tok[0]) - 1, int(tok[1]) - 1, int(tok[2])))

    return elements, np.array(positions, dtype=float), bonds


def get_molecule(name_or_cid: str | int) -> Atoms:
    """Return an ASE Atoms for a PubChem compound, with bonds + provenance in .info."""
    cid = name_or_cid if isinstance(name_or_cid, int) else name_to_cid(name_or_cid)
    elements, positions, bonds = parse_sdf(fetch_sdf(cid))
    atoms = Atoms(symbols=elements, positions=positions)
    atoms.info["bonds"] = bonds
    atoms.info["provenance"] = {
        "source": "PubChem",
        "cid": cid,
        "query": name_or_cid,
        "n_atoms": len(elements),
    }
    return atoms
