"""Backbone end-to-end demo — the canonical CNT + methanol walkthrough, headless.

PARSE -> GROUND (KG evidence + gaps) -> PROPOSE (Gate 1) -> BUILD & SHOW every
constituent (Gate 2 sandboxed exec, Gate 3 verify) -> SHOWCASE assembly.

    conda activate mdagent && python scripts/backbone_demo.py [--no-llm]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

if "--no-llm" in sys.argv:
    from mtagent import llm
    llm.get_openai_key = lambda: None

from ase.io import write as ase_write

from mtagent import clarify, ground
from mtagent.execute import run_snippet
from mtagent.propose import propose
from mtagent.verify import verify_atoms

QUERY = "a (6,6) carbon nanotube, 10 unit cells long, with methanol inside"
OUT = Path("data/out/backbone_demo")


def main() -> None:
    print(f"QUERY: {QUERY}\n")

    print("== PARSE ==")
    state = clarify.parse_query(QUERY)
    for c in state["constituents"]:
        print(f"  {c['key']}: {c['builder']} {c['spec']}")
    print(f"  relation: {state['relation']}\n")

    print("== GAP CHECK (registry-derived; answering 'default' to each) ==")
    for _ in range(10):
        gs = ground.gaps(state)
        if not gs:
            break
        print(f"  Q: {gs[0].question.splitlines()[0]}\n  A: default")
        state = clarify.apply_answer(state, "default", gs[0])
    final = ground.finalize(state)
    print(f"  spec complete: {json.dumps(final['constituents'])}\n")

    OUT.mkdir(parents=True, exist_ok=True)
    artifacts = {}
    for c in final["constituents"]:
        print(f"== {c['key']}: GROUND -> PROPOSE -> BUILD ==")
        ev = ground.evidence_for(c)
        for s in ev.signatures:
            print(f"  KG evidence: {s.split('#')[0].strip()}")
        p = propose(ev, f"Build '{c['key']}': {c['builder']} spec {json.dumps(c['spec'])}")
        print(f"  proposal [{p.source}] — {p.report.summary()}")
        print("  " + p.code.replace("\n", "\n  "))
        atoms = run_snippet(p.code)                                  # Gate 2
        rep = verify_atoms(atoms)                                    # Gate 3
        print(f"  built: {len(atoms)} atoms — Gate 3 {rep.summary()}")
        ase_write(OUT / f"{c['key']}.xyz", atoms)
        artifacts[c["key"]] = atoms
        print()

    if final.get("relation"):
        rel = final["relation"]
        print(f"== SHOWCASE: {rel['guest']} {rel['kind']} {rel['host']} ==")
        ev = ground.evidence_for_relation(final)
        p = propose(ev, f"Combine: {rel['guest']} {rel['kind']} {rel['host']} "
                        "(both are already-built ase.Atoms variables).")
        print(f"  proposal [{p.source}] — {p.report.summary()}")
        print("  " + p.code.replace("\n", "\n  "))
        showcase = run_snippet(p.code, artifacts)
        rep = verify_atoms(showcase)
        print(f"  showcase: {len(showcase)} atoms — Gate 3 {rep.summary()}")
        print(f"  {showcase.info.get('assembly')}")
        ase_write(OUT / "showcase.xyz", showcase)
        print("\nNOTE: the showcase cell is geometry only — a simulation-ready cell "
              "needs MD equilibration (out of scope for the Moltemplate agent).")
    print(f"\nwrote .xyz files to {OUT}/")


if __name__ == "__main__":
    main()
