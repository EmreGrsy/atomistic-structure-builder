"""Tests for the Moltemplate KG: retrieval + Gate 1 validator.

Run: python tests/test_kg.py   (plain asserts, no pytest required)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent.kg import MoltemplateKG

kg = MoltemplateKG()
passed = 0


def check(name: str, cond: bool) -> None:
    global passed
    assert cond, f"FAILED: {name}"
    passed += 1
    print(f"  ok  {name}")


# --- retrieval ---------------------------------------------------------------
r = kg.retrieve("gold nanoparticle coated with oleic acid")
check("geometry_only active by default", "geometry_only" in r.conditions)
check("rigid_core triggered by 'nanoparticle'", "rigid_core" in r.conditions)
check("reusable_placement triggered by 'coated'", "reusable_placement" in r.conditions)
check("force-field sections excluded", "In Settings" in r.excluded)
check("Data Atoms allowed", "Data Atoms" in r.allowed)
check("move/rot allowed for placement", {"move", "rot"} <= r.allowed)

r2 = kg.retrieve("build a crystal lattice array of molecules")
check("regular_array triggered by 'array'/'lattice'", "regular_array" in r2.conditions)

# --- Gate 1: our real emitted files must PASS --------------------------------
work = Path(__file__).resolve().parent.parent / "data" / "work"
for fname in ("mol.lt", "core.lt", "system.lt"):
    f = work / fname
    if f.exists():
        rep = kg.validate_lt(f.read_text())
        check(f"emitted {fname} passes Gate 1", rep.passed)

# --- Gate 1: hallucinations must be caught -----------------------------------
bad_transform = 'w1 = new Mol.rotate(90,0,0,1).move(1,2,3)'
rep = kg.validate_lt(bad_transform)
check("catches hallucinated .rotate", not rep.passed)
check("suggests .rot for .rotate",
      any(i.suggestion == "rot" for i in rep.errors()))

bad_args = 'w1 = new Mol.rot(90,0,0).move(1,2,3)'   # rot needs 4 or 7 args
rep = kg.validate_lt(bad_args)
check("catches wrong arg count on .rot (3)", not rep.passed)

# the false-positive fix: 7-arg .rot (rotate about a point) is VALID
rep = kg.validate_lt('w = new Mol.rot(90,0,0,1,5,5,5).move(1,2,3)')
check("accepts 7-arg .rot (about a point)", rep.passed)
rep = kg.validate_lt('w = new Mol.scale(2,2,2,0,0,0)')      # 6-arg scale valid
check("accepts 6-arg .scale", rep.passed)
rep = kg.validate_lt('w = new Mol.matrix(1,0,0,0,1,0,0,0,1)')
check("accepts 9-arg .matrix", rep.passed)
rep = kg.validate_lt('@{atom:type.rjust(3)}')               # formatting, not a transform
check("does not flag .rjust formatting method", rep.passed)

bad_section = 'write("Data Atomz") { }'
rep = kg.validate_lt(bad_section)
check("catches misspelled section 'Data Atomz'", not rep.passed)
check("suggests 'Data Atoms'",
      any(i.suggestion == "Data Atoms" for i in rep.errors()))

good = '''Mol {
  write_once("Data Masses") { @atom:O 15.999 }
  write("Data Atoms") { $atom:a0 $mol:. @atom:O 0.0 0 0 0 }
  write("Data Bonds") { $bond:b0 @bond:OH $atom:a0 $atom:a0 }
}
c = new Mol.rot(90,0,0,1).move(1,2,3)'''
rep = kg.validate_lt(good)
check("clean .lt passes with no errors", rep.passed)

# force-field section in geometry-only mode -> warning, not error
rep = kg.validate_lt('write("Data Bond Coeffs") { }')
check("force-field section warns but does not fail", rep.passed and len(rep.issues) == 1)

# custom section (manual §5.6) -> warning, not error; near-typo -> error
rep = kg.validate_lt('write_once("Data Foo Fee Fum") { }')
check("custom section warns, not fails", rep.passed)
rep = kg.validate_lt('write("Data Atomz") { }')
check("near-typo section still errors", not rep.passed)

print(f"\nALL {passed} CHECKS PASSED")
