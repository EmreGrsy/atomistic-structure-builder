"""Demo the ASE KG (introspected) + Gate 1 validator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent.ase_kg import ASEKnowledgeGraph
from mtagent.ase_validate import ASEValidator

kg = ASEKnowledgeGraph()
val = ASEValidator(kg)

print("=== ASE KG stats (introspected from installed ase) ===")
print(kg.stats())

print("\n=== retrieve 'optimize' ===")
for e in kg.retrieve("optimize", 5):
    print(f"  {e.qualname:35} {e.doc[:55]}")

print("\n=== Gate 1 on valid ASE code ===")
print(val.validate("""
from ase.build import bulk
from ase.io import write
a = bulk('Au', 'fcc', a=4.078, cubic=True)
write('au.pdb', a)
""").summary())

print("\n=== Gate 1 on hallucinated ASE code ===")
print(val.validate("""
from ase.build import bulk
a = bulk('Au', crystalstruct='fcc')      # wrong kwarg name
import ase.build
ase.build.nanoparticle('Au')             # no such function
""").summary())
