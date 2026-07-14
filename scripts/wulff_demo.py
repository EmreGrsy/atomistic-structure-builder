"""Build a Wulff magnetite NP and validate against the user's existing 4nm data."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from ase.io import write

from mtagent.wulff import build_magnetite_wulff
from mtagent.runner import parse_lammps_data

GROUND_TRUTH = "/home/emre/Desktop/23021_oa_on_nanosphere/3_nsp_hydroxylation/magnetite_4nm_swp.data"


def main() -> None:
    print("Building Wulff magnetite NP (target 40 A) ...")
    np_atoms = build_magnetite_wulff(diameter=40.0)
    prov = np_atoms.info["provenance"]
    comp = prov["composition"]
    fe, o = comp.get("Fe", 0), comp.get("O", 0)
    ext = np_atoms.get_positions().max(0) - np_atoms.get_positions().min(0)
    print(f"  built: {len(np_atoms)} atoms  Fe:O={fe}:{o}  Fe/O={fe/max(o,1):.3f}  "
          f"extent={ext.round(1)}  facets={prov['facets']}")

    if Path(GROUND_TRUTH).exists():
        els, pos = parse_lammps_data(GROUND_TRUTH)
        cc = Counter(els)
        gext = pos.max(0) - pos.min(0)
        gfe, go = cc.get("Fe", 0), cc.get("O", 0)
        print(f"  yours: {len(els)} atoms  Fe:O={gfe}:{go}  Fe/O={gfe/max(go,1):.3f}  "
              f"extent={gext.round(1)}")

    out = Path("data/out")
    out.mkdir(parents=True, exist_ok=True)
    write(str(out / "magnetite_wulff.xyz"), np_atoms)
    write(str(out / "magnetite_wulff.png"), np_atoms,
          rotation="15x,10y,0z", radii=0.55, scale=12)
    print(f"\n  wrote {out}/magnetite_wulff.xyz and .png")


if __name__ == "__main__":
    main()
