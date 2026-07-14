"""Demo the Moltemplate KG: stats, retrieval, and Gate 1 on good vs hallucinated .lt."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent.kg import MoltemplateKG

kg = MoltemplateKG()

print("=== KG stats ===")
print(kg.stats())

print("\n=== Retrieval: 'gold nanoparticle coated with oleic acid' ===")
r = kg.retrieve("gold nanoparticle coated with oleic acid")
print("conditions:", r.conditions)
print("allowed   :", sorted(r.allowed))
print("excluded  :", sorted(r.excluded))
for n in r.notes:
    print("  -", n)

print("\n=== Gate 1 on our real emitted system.lt ===")
sysf = Path("data/work/system.lt")
if sysf.exists():
    print(kg.validate_lt(sysf.read_text()).summary())

print("\n=== Gate 1 on a hallucinated snippet ===")
bad = 'w1 = new Mol.rotate(90,0,0,1).move(1,2)\nwrite("Data Atomz") { }'
print("input:", repr(bad))
print(kg.validate_lt(bad).summary())
