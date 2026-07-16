"""Derive the bundled MOF frameworks in data/mofs/ from their published source.

The MOF CIFs this agent builds from are NOT hand-typed: they come from CoRE MOF
(computation ready, solvent removed from the pores) and are checked against the
literature before they are written. This script records exactly how each file
was produced so it can be re-derived and audited.

Usage (the source zip is ~81 MB, so it is not vendored):

    curl -sL -o core2019.zip "https://zenodo.org/records/14184621/files/CoREMOF2019_public_v2_20241119.zip?download=1"
    python scripts/prepare_mofs.py core2019.zip

WHY A REPAIR STEP IS NEEDED (ZIF-8): the CoRE MOF entry VELVOY_clean_pacman
carries the methyl hydrogens of every 2-methylimidazolate linker at BOTH
refined rotamer positions with full occupancy. That is 8 H per linker instead
of 5 (formula Zn12C96H192N48, 348 atoms) and puts H atoms 0.79 A apart, which
is why CoRE MOF flags it "not computation ready". Each methyl's 6 H sit in 3
close pairs, one H per pair per rotamer, so keeping one H from each pair
recovers a real CH3 and the published Zn12C96H120N48 (276 atoms). The kept
rotamer is verified tetrahedral (H-C-H ~109.5 deg) afterwards.
"""
from __future__ import annotations

import itertools
import sys
import zipfile
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "mofs"

# entry in the CoRE MOF 2019 zip -> bundled file + what the literature says it
# must come out as (checked, not trusted)
SOURCES = {
    "ZIF-8.cif": dict(
        member="CoREMOF2019_public_v2_20241118/NCR/ASR/both/VELVOY_clean_pacman.cif",
        formula="C96H120N48Zn12",
        a=16.9910,
        density=0.924,          # g/cm3, Park et al. PNAS 103, 10186 (2006)
        zn_n=1.99,              # A, tetrahedral Zn-N
        repair="methyl_rotamers",
        note=("ZIF-8, zinc 2-methylimidazolate, sodalite topology. Source: "
              "CoRE MOF 2019 entry VELVOY_clean_pacman (all solvent removed), "
              "https://doi.org/10.5281/zenodo.14184621. Structure first "
              "reported by Park et al., PNAS 103, 10186 (2006), CSD VELVOY. "
              "Disordered methyl hydrogens resolved to one rotamer by "
              "scripts/prepare_mofs.py; PACMAN charges dropped (geometry "
              "only). Framework atoms only: the pores are empty."),
    ),
}


def _resolve_methyl_rotamers(atoms: Atoms) -> Atoms:
    """Keep one H of each disordered methyl pair, recovering a real CH3."""
    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 999.0)
    sym = np.array(atoms.get_chemical_symbols())
    H = np.where(sym == "H")[0]
    drop: set[int] = set()
    for c in np.where(sym == "C")[0]:
        hs = [h for h in H if d[c, h] < 1.25]
        if len(hs) != 6:            # 1 = ring CH, 3 = ordered methyl, 0 = ring C
            continue
        kept: list[int] = []
        for h in hs:
            if any(d[h, k] < 1.0 for k in kept):   # the partner of a kept H
                drop.add(h)
            else:
                kept.append(h)
        if len(kept) != 3:
            raise ValueError(f"methyl at C{c} did not split into 3 pairs")
    return atoms[[i for i in range(len(atoms)) if i not in drop]]


def _verify(atoms: Atoms, spec: dict, name: str) -> None:
    """Refuse to bundle anything that does not match the published structure."""
    got = atoms.get_chemical_formula()
    if got != spec["formula"]:
        raise ValueError(f"{name}: formula {got} != published {spec['formula']}")
    if abs(atoms.cell.lengths()[0] - spec["a"]) > 0.01:
        raise ValueError(f"{name}: a = {atoms.cell.lengths()[0]:.4f} != {spec['a']}")

    d = atoms.get_all_distances(mic=True)
    np.fill_diagonal(d, 999.0)
    sym = np.array(atoms.get_chemical_symbols())
    heavy = np.where(sym != "H")[0]
    if d[np.ix_(heavy, heavy)].min() < 1.2:
        raise ValueError(f"{name}: heavy atoms {d[np.ix_(heavy, heavy)].min():.2f} A apart")

    rho = atoms.get_masses().sum() / 6.02214076e23 / (atoms.get_volume() * 1e-24)
    if abs(rho - spec["density"]) > 0.01:
        raise ValueError(f"{name}: density {rho:.3f} != published {spec['density']}")

    Zn, N = np.where(sym == "Zn")[0], np.where(sym == "N")[0]
    if len(Zn):
        coord = [(d[z, N] < 2.3).sum() for z in Zn]
        if set(coord) != {4}:
            raise ValueError(f"{name}: Zn coordination {set(coord)} != 4")
        zn_n = d[np.ix_(Zn, N)]
        if abs(zn_n[zn_n < 2.3].mean() - spec["zn_n"]) > 0.05:
            raise ValueError(f"{name}: Zn-N {zn_n[zn_n < 2.3].mean():.3f} off")

    for c in np.where(sym == "C")[0]:      # every methyl must be tetrahedral
        hs = [h for h in np.where(sym == "H")[0] if d[c, h] < 1.25]
        if len(hs) != 3:
            continue
        for i, j in itertools.combinations(hs, 2):
            ang = atoms.get_angle(i, c, j, mic=True)
            if not 104 < ang < 115:
                raise ValueError(f"{name}: H-C-H {ang:.1f} deg is not tetrahedral")


def main(zip_path: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    z = zipfile.ZipFile(zip_path)
    tmp = OUT / "_src.cif"
    for name, spec in SOURCES.items():
        tmp.write_bytes(z.read(spec["member"]))
        atoms = read(str(tmp))
        if spec.get("repair") == "methyl_rotamers":
            atoms = _resolve_methyl_rotamers(atoms)
        _verify(atoms, spec, name)
        atoms.set_pbc(True)
        path = OUT / name
        write(str(path), atoms, format="cif")
        path.write_text("# " + spec["note"].replace(". ", ".\n# ") + "\n"
                        + path.read_text())
        print(f"{name}: {atoms.get_chemical_formula()} "
              f"({len(atoms)} atoms) verified and written to {path}")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
