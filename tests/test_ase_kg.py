"""Tests for the ASE KG + Gate 1 validator.  Run: python tests/test_ase_kg.py"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent.ase_kg import ASEKnowledgeGraph
from mtagent.ase_validate import ASEValidator

kg = ASEKnowledgeGraph()
val = ASEValidator(kg)
passed = 0


def check(name: str, cond: bool) -> None:
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


# --- KG introspection --------------------------------------------------------
check("KG has many entries", len(kg.entries) > 500)
check("ase.build.bulk present", kg.get("ase.build.bulk") is not None)
check("bulk knows 'crystalstructure' kwarg", "crystalstructure" in kg.get("ase.build.bulk").keyword_names)
check("ase.Atoms present", kg.get("ase.Atoms") is not None)
check("Atoms methods captured", "get_positions" in kg.atoms_methods)
check("retrieval finds cluster builders", any("cluster" in e.qualname.lower()
      for e in kg.retrieve("cluster")))

# --- Gate 1: valid code passes ----------------------------------------------
ok_code = """
from ase.build import bulk
from ase.io import read, write
a = bulk('Au', 'fcc', a=4.078, cubic=True)
atoms = read('in.xyz')
write('out.pdb', atoms)
"""
check("valid ASE code passes", val.validate(ok_code).passed)

# --- Gate 1: hallucinated keyword arg ---------------------------------------
r = val.validate("from ase.build import bulk\nb = bulk('Au', crystalstruct='fcc')")
check("catches hallucinated kwarg 'crystalstruct'", not r.passed)
check("suggests 'crystalstructure'",
      any(i.suggestion == "crystalstructure" for i in r.issues))

# --- Gate 1: unknown ASE function -------------------------------------------
r = val.validate("import ase.build\nase.build.nanoparticle('Au')")
check("catches unknown ase.build.nanoparticle", not r.passed)

# --- Gate 1: too many positional args ---------------------------------------
r = val.validate("from ase.build import bulk\nbulk('Au','fcc',4,4,4,4,4)")
check("catches too many positional args", not r.passed)

# --- Gate 1: Atoms constructor -----------------------------------------------
r = val.validate("from ase import Atoms\nAtoms(symbols='H2O', positions=[[0,0,0]])")
check("valid Atoms constructor passes", r.passed)
# KNOWN LIMITATION: ase.Atoms is (symbols=None, *args, calculator=None, **kwargs),
# so its signature legitimately accepts any kwarg -> bad kwargs are NOT catchable here.
r = val.validate("from ase import Atoms\nAtoms(symbolz='H2O')")
check("Atoms(**kwargs) is permissive by design (known limitation)", r.passed)

# --- Gate 1: method calls on variables are skipped (no false positive) -------
r = val.validate("from ase.build import bulk\na = bulk('Cu')\ne = a.get_potential_energy()")
check("method call on variable not falsely flagged", r.passed)

# --- Gate 1: syntax error caught --------------------------------------------
check("syntax error caught", not val.validate("from ase.build import (").passed)

print(f"\nALL {passed} CHECKS PASSED")
