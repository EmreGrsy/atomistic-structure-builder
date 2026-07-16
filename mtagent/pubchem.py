"""Fetch molecular structures from PubChem (provenance-tracked, cached).

Anti-hallucination axis #1: coordinates come from a real database, never invented.
Every returned structure carries ``atoms.info['provenance']`` so downstream verifiers
(Gate 3) can confirm every atom traces back to a source.
"""
from __future__ import annotations

import json
from difflib import SequenceMatcher
from pathlib import Path
from urllib.parse import quote

import numpy as np
import requests
from ase import Atoms

_REST = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
_AUTOCOMPLETE = "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound"
_CACHE = Path(__file__).resolve().parent.parent / "data" / "cache"


class UnknownCompound(ValueError):
    """PubChem has no compound under the given name.

    Carries `.query` and `.suggestions` so callers can offer alternatives
    instead of dumping a raw HTTP error at the user.
    """

    def __init__(self, query: str, suggestions: list[str] | None = None):
        self.query = query
        self.suggestions = suggestions or []
        msg = f"PubChem has no molecule named {query!r}."
        if self.suggestions:
            msg += (" Names starting the same way: "
                    + ", ".join(self.suggestions)
                    + ". If you meant one of those, say it by name.")
        else:
            msg += " Check the spelling, or give the PubChem CID as a number."
        super().__init__(msg)


def _cache_dir() -> Path:
    _CACHE.mkdir(parents=True, exist_ok=True)
    return _CACHE


def _name_cache() -> Path:
    return _cache_dir() / "name_to_cid.json"


def _info_cache() -> Path:
    return _cache_dir() / "cid_info.json"


def _read_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _write_cache(path: Path, data: dict) -> None:
    try:
        path.write_text(json.dumps(data, indent=1, sort_keys=True))
    except Exception:
        pass                      # cache is an optimisation, never a gate


def suggest_names(name: str, limit: int = 5) -> list[str]:
    """Compound names PubChem knows that START with `name` (best effort).

    PubChem's autocomplete is a PREFIX matcher, not a spell checker: a typo
    like 'cannoaboild' returns 'Cannogenin thevetoside', a different molecule
    entirely. So these are only ever shown to the user as candidates, never
    substituted for what they asked for.
    """
    try:
        r = requests.get(f"{_AUTOCOMPLETE}/{quote(str(name))}/json",
                         params={"limit": max(limit, 5)}, timeout=15)
        r.raise_for_status()
        hits = list(r.json().get("dictionary_terms", {}).get("compound", []))
    except Exception:
        return []
    # a shared prefix alone is weak evidence ('unobtainium' -> 'Carcainium
    # (chloride)'), so keep only names that actually look like what was typed
    q = str(name).strip().lower()
    close = [(SequenceMatcher(None, q, h.lower()).ratio(), h) for h in hits]
    return [h for score, h in sorted(close, reverse=True)
            if score >= 0.6][:limit]


def name_to_cid(name: str) -> int:
    """Resolve a compound name to its PubChem CID (cached).

    Raises `UnknownCompound` (not a bare HTTPError) when the name is unknown,
    so the app can explain it and suggest alternatives.
    """
    key = str(name).strip().lower()
    if key.isdigit():             # a bare number IS a CID (we tell users so)
        return int(key)
    cache = _read_cache(_name_cache())
    if key in cache:
        return int(cache[key])
    r = requests.get(f"{_REST}/compound/name/{quote(str(name))}/cids/TXT",
                     timeout=30)
    if r.status_code == 404:
        raise UnknownCompound(str(name), suggest_names(name))
    r.raise_for_status()
    cid = int(r.text.strip().splitlines()[0])
    cache[key] = cid
    _write_cache(_name_cache(), cache)
    return cid


def compound_info(cid: int) -> dict:
    """Canonical title + molecular formula for a CID (cheap, no 3D download)."""
    cache = _read_cache(_info_cache())
    key = str(int(cid))
    info = cache.get(key)
    if info is None:
        r = requests.get(f"{_REST}/compound/cid/{int(cid)}/property/"
                         f"MolecularFormula,Title/JSON", timeout=30)
        r.raise_for_status()
        p = r.json()["PropertyTable"]["Properties"][0]
        info = {"cid": int(cid), "title": p.get("Title", ""),
                "formula": p.get("MolecularFormula", "")}
    # a CID that does not exist still answers 200, echoing back just the CID
    # with no properties — that empty record IS the "not found". Checked on
    # the cached path too, so an empty record cached by an older build cannot
    # keep masquerading as a real compound.
    if not info.get("title") and not info.get("formula"):
        cache.pop(key, None)
        _write_cache(_info_cache(), cache)
        raise ValueError(f"PubChem has no compound with CID {int(cid)}.")
    cache[key] = info
    _write_cache(_info_cache(), cache)
    return info


def fetch_sdf(cid: int) -> str:
    """Download the 3D SDF record for a CID, caching to data/cache."""
    path = _cache_dir() / f"cid_{cid}_3d.sdf"
    if path.exists():
        return path.read_text()
    r = requests.get(f"{_REST}/compound/cid/{cid}/record/SDF",
                     params={"record_type": "3d"}, timeout=60)
    if r.status_code == 404:
        # either the CID does not exist, or PubChem holds no 3D conformer for
        # it (common for very large or floppy molecules) — say which
        if not _cid_exists(cid):
            raise ValueError(f"PubChem has no compound with CID {cid}.")
        raise ValueError(
            f"PubChem has no 3D structure for CID {cid}. This agent places "
            f"real 3D coordinates, so a molecule with no 3D record cannot be "
            f"built. A smaller or more common molecule will have one.")
    r.raise_for_status()
    path.write_text(r.text)
    return r.text


def _cid_exists(cid: int) -> bool:
    try:
        compound_info(cid)
        return True
    except Exception:
        return False


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
    cid = name_or_cid if isinstance(name_or_cid, int) \
        else name_to_cid(name_or_cid)      # resolves bare digits as a CID too
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
