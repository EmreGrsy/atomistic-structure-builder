"""Hydroxylate the Wulff magnetite NP + validate coordination logic against user data."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ase import Atoms
from ase.io import write

from mtagent.wulff import build_magnetite_wulff
from mtagent.hydroxylate import hydroxylate, coordination_numbers, select_undercoordinated
from mtagent.runner import parse_lammps_data

GT = "/home/emre/Desktop/23021_oa_on_nanosphere/3_nsp_hydroxylation/"


def main() -> None:
    print("=== Hydroxylate Wulff magnetite NP (46 A) ===")
    npart = build_magnetite_wulff(diameter=46.0)
    hx = hydroxylate(npart, cation="Fe", anion="O", cutoff=None, threshold=2)
    info = hx.info["hydroxylation"]
    print(f"  cations: {info['n_cations']}, coord dist: {info['coord_distribution']}")
    print(f"  inferred bulk coordination: {info['bulk_coordination']}")
    print(f"  hydroxylated {info['n_undercoordinated']} sites → +{2*info['n_undercoordinated']} atoms, "
          f"{len(info['oh_bonds'])} O-H bonds")
    write("data/out/magnetite_hydroxylated.xyz", hx)
    write("data/out/magnetite_hydroxylated.png", hx, rotation="15x,10y,0z", radii=0.5, scale=12)

    print("\n=== Validate coordination logic vs your magnetite_4nm_swp.data (cutoff 3.0) ===")
    p = Path(GT) / "magnetite_4nm_swp.data"
    if p.exists():
        els, pos = parse_lammps_data(str(p))
        gt = Atoms(symbols=els, positions=pos)
        coord = coordination_numbers(gt, "Fe", "O", cutoff=3.0)
        under = select_undercoordinated(coord, threshold=2)
        print(f"  your NP: {len(gt)} atoms, coord dist {dict(sorted(Counter(coord.values()).items()))}")
        print(f"  undercoordinated Fe (<=2 O): {len(under)}   (your TCL capped hydroxylate_num=11)")
        # your hydroxylated file added how many atoms?
        hp = Path(GT) / "magnetite_4nm_hydroxylated.data"
        if hp.exists():
            hels, _ = parse_lammps_data(str(hp))
            print(f"  your hydroxylated file: {len(hels)} atoms (+{len(hels)-len(els)} vs swp)")

    print("\n=== Genericity check: same analyzer on a Cu2O-like pair ===")
    from ase.build import bulk as _bulk
    cu = _bulk("Cu", "fcc", a=3.615, cubic=True).repeat((3, 3, 3))
    c2 = coordination_numbers(cu, "Cu", "Cu", cutoff=3.0)
    print(f"  Cu fcc (Cu-Cu within 3.0): dist {dict(sorted(Counter(c2.values()).items()))} — runs on any composition")


if __name__ == "__main__":
    main()
