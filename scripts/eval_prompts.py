"""Evaluation harness: N realistic prompts through the FULL pipeline.

Stages per prompt (each recorded independently):
  parse     — LLM parse produced >=1 registry constituent
  spec      — gaps answered with 'default', finalize() clean
  gate1     — a proposal exists for every constituent (+ showcase) and passed
              static KG validation
  build     — every constituent snippet executed (sandboxed)
  checks    — Gate 3 geometry checks all pass
  assemble  — showcase snippet executed + checked (when relations exist)

Usage:  python scripts/eval_prompts.py [--limit N] [--out data/out/eval]
Writes results.jsonl (one record per prompt) + summary.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

PROMPTS = [
    # --- metal nanoparticles ------------------------------------------------
    ("metal_np", "a 3 nm gold nanoparticle"),
    ("metal_np", "silver nanoparticle, 25 angstrom diameter"),
    ("metal_np", "a 2 nm platinum nanocube"),
    ("metal_np", "copper nanoparticle"),
    ("metal_np", "a palladium nanoparticle of 30 A"),
    ("metal_np", "nickel nanoparticle 2.5 nm"),
    ("metal_np", "a titanium nanoparticle, 24 angstrom"),
    ("metal_np", "aluminum nanoparticle with diameter 20 A"),
    ("metal_np", "a 2.8 nm silver cube"),
    ("metal_np", "gold nanoparticle, sphere, 18 angstrom"),
    # --- oxide nanoparticles ------------------------------------------------
    ("oxide_np", "a 3 nm magnetite nanoparticle"),
    ("oxide_np", "magnetite nanoparticle, wulff shape, 2.5 nm"),
    ("oxide_np", "a magnetite nanoparticle with only 111 and 100 facets, 3 nm"),
    ("oxide_np", "magnetite nanosphere of 28 angstrom"),
    ("oxide_np", "a 2 nm magnetite cube"),
    ("oxide_np", "iron oxide nanoparticle 3 nm"),
    ("oxide_np", "Fe3O4 nanoparticle, 26 A, wulff"),
    ("oxide_np", "magnetite particle where the 110 surface energy is 1.2, 3 nm"),
    ("oxide_np", "a faceted magnetite nanoparticle with gamma_100 of 1.1, 30 A"),
    ("oxide_np", "magnetite nanoparticle 2 nm with no 110 facet"),
    # --- solvation ----------------------------------------------------------
    ("solvation", "a 2 nm magnetite nanoparticle in water"),
    ("solvation", "gold nanoparticle, 2 nm, solvated in ethanol"),
    ("solvation", "a 20 A silver nanoparticle in a water box"),
    ("solvation", "magnetite nanoparticle 2.5 nm surrounded by methanol"),
    ("solvation", "a copper nanoparticle, 18 A, in water"),
    ("solvation", "2 nm platinum nanoparticle immersed in water"),
    ("solvation", "a small magnetite cube in ethanol, 20 A"),
    ("solvation", "nickel nanoparticle 2 nm inside a box of water"),
    ("solvation", "a 25 A gold sphere in toluene"),
    ("solvation", "magnetite wulff nanoparticle, 24 A, in hexane"),
    # --- elemental slabs ----------------------------------------------------
    ("element_slab", "a gold 111 surface"),
    ("element_slab", "aluminum 001 slab"),
    ("element_slab", "copper 110 surface, 3x3"),
    ("element_slab", "an iron 110 slab"),
    ("element_slab", "platinum 557 surface"),
    ("element_slab", "titanium 0001 surface"),
    ("element_slab", "a silicon 100 slab"),
    ("element_slab", "germanium 100 surface"),
    ("element_slab", "a tungsten 110 surface, 4x4 supercell"),
    ("element_slab", "silver 100 slab with 12 angstrom vacuum"),
    # --- compound slabs -----------------------------------------------------
    ("compound_slab", "a rutile TiO2 110 surface"),
    ("compound_slab", "anatase 101 slab"),
    ("compound_slab", "alpha quartz 0001 surface"),
    ("compound_slab", "a zinc oxide 0001 surface"),
    ("compound_slab", "GaAs 110 slab"),
    ("compound_slab", "an MgO 100 surface"),
    ("compound_slab", "SrTiO3 100 slab"),
    ("compound_slab", "sodium chloride 100 surface"),
    ("compound_slab", "an alumina 0001 slab"),
    ("compound_slab", "hematite 0001 surface"),
    # --- solid-liquid interfaces (coats) -------------------------------------
    ("interface", "water on an aluminum 111 surface"),
    ("interface", "30 water molecules on a gold 111 slab"),
    ("interface", "ethanol on a magnetite 001 surface"),
    ("interface", "a silicon 100 surface covered with water"),
    ("interface", "15 methanol molecules on a platinum 111 surface"),
    ("interface", "water on rutile TiO2 110"),
    ("interface", "10 oleic acid molecules on a gold 111 slab"),
    ("interface", "a copper 111 surface coated with 40 waters"),
    ("interface", "ethanol layer on quartz 0001"),
    ("interface", "2 water molecules placed on an MgO 100 surface"),
    # --- sandwiches / confinement -------------------------------------------
    ("sandwich", "water between two gold 111 slabs"),
    ("sandwich", "30 water molecules sandwiched between two magnetite 001 slabs"),
    ("sandwich", "water between two graphene sheets"),
    ("sandwich", "ethanol confined between two aluminum 111 slabs"),
    ("sandwich", "water between a magnetite 001 slab and a rutile 110 slab"),
    ("sandwich", "a water film between two silicon 100 surfaces"),
    ("sandwich", "40 waters between two copper 111 slabs"),
    ("sandwich", "methanol between two graphene layers"),
    ("sandwich", "water sandwiched between MgO 100 and NaCl 100 slabs"),
    ("sandwich", "20 ethanol molecules between two quartz 0001 slabs"),
    # --- supercrystals / clusters -------------------------------------------
    ("supercrystal", "a supercrystal of 4 magnetite nanoparticles, 2 nm each"),
    ("supercrystal", "FCC supercrystal of 4 gold nanoparticles, 18 A"),
    ("supercrystal", "a cluster of 8 magnetite nanoparticles, 2 nm"),
    ("supercrystal", "bcc superlattice of 2 magnetite nanoparticles, 20 A"),
    ("supercrystal", "magnetite supercrystal made of magnetite nanoparticles, use FCC"),
    ("supercrystal", "a cluster of 3 silver nanoparticles with 15 A gaps, 2 nm each"),
    ("supercrystal", "4 copper nanoparticles arranged in a cluster, 18 A each"),
    ("supercrystal", "fcc superlattice of 4 magnetite cubes, 18 A"),
    ("supercrystal", "a dimer of two 2 nm gold nanoparticles"),
    ("supercrystal", "supercrystal with 6 magnetite nanoparticles, 20 A diameter"),
    # --- nanotubes / filling -------------------------------------------------
    ("nanotube", "a (6,6) carbon nanotube, 10 cells long"),
    ("nanotube", "carbon nanotube with methanol inside"),
    ("nanotube", "a (10,10) CNT filled with 8 ethanol molecules"),
    ("nanotube", "a zigzag (9,0) carbon nanotube"),
    ("nanotube", "water inside a (8,8) carbon nanotube"),
    ("nanotube", "a carbon nanotube, 12 unit cells, with 5 waters inside"),
    ("nanotube", "(7,7) CNT with acetone inside"),
    ("nanotube", "a short (6,6) nanotube, 6 cells"),
    ("nanotube", "carbon nanotube containing 10 methanol molecules"),
    ("nanotube", "a (12,0) carbon nanotube with water inside"),
    # --- bulk / molecules / boxes / sheets -----------------------------------
    ("other", "magnetite bulk, 2x2x2"),
    ("other", "gold bulk crystal 3x3x3"),
    ("other", "bulk silicon"),
    ("other", "an MgO bulk crystal"),
    ("other", "a single oleic acid molecule"),
    ("other", "a caffeine molecule"),
    ("other", "a 25 angstrom box of water"),
    ("other", "an ethanol solvent box, 30 A"),
    ("other", "a graphene sheet"),
    ("other", "an h-BN sheet"),
]


def run_one(category: str, prompt: str) -> dict:
    from mtagent import clarify, ground
    from mtagent.execute import run_snippet
    from mtagent.propose import propose
    from mtagent.verify import verify_atoms

    rec = {"category": category, "prompt": prompt, "stage": "parse",
           "ok": False, "error": None, "natoms": None, "seconds": None}
    t0 = time.time()
    try:
        state = clarify.parse_query(prompt)
        if not state.get("constituents"):
            raise RuntimeError("no constituents parsed")
        rec["builders"] = [c["builder"] for c in state["constituents"]]

        rec["stage"] = "spec"
        for _ in range(8):
            gs = ground.gaps(state)
            if not gs:
                break
            state = clarify.apply_answer(state, "default", gs[0])
        final = ground.finalize(state)
        rels = ground.relations_of(final)
        rec["relations"] = [r["kind"] for r in rels]

        rec["stage"] = "gate1"
        proposals = {}
        for c in final["constituents"]:
            p = propose(ground.evidence_for(c),
                        f"Build '{c['key']}': {c['builder']} with spec "
                        f"{json.dumps(c['spec'])}")
            if not p.report.passed:
                raise RuntimeError(f"gate1 failed for {c['key']}: {p.report.summary()}")
            proposals[c["key"]] = p
        if rels:
            desc = "; then ".join(
                f"{r['guest']} {r['kind'].replace('_', ' ')} {r['host']} "
                f"(params {json.dumps(r.get('params') or {})})" for r in rels)
            p = propose(ground.evidence_for_relation(final),
                        f"Combine, in order: {desc}. All constituent variables "
                        "are already-built ase.Atoms.")
            if not p.report.passed:
                raise RuntimeError("gate1 failed for showcase")
            proposals["showcase"] = p

        rec["stage"] = "build"
        built = {}
        for c in final["constituents"]:
            built[c["key"]] = run_snippet(proposals[c["key"]].code)

        rec["stage"] = "checks"
        for key, atoms in built.items():
            rep = verify_atoms(atoms)
            if not rep.passed:
                raise RuntimeError(f"gate3 failed for {key}: {rep.summary()}")

        if rels:
            rec["stage"] = "assemble"
            atoms = run_snippet(proposals["showcase"].code, dict(built))
            rep = verify_atoms(atoms)
            if not rep.passed:
                raise RuntimeError(f"gate3 failed for showcase: {rep.summary()}")
            rec["natoms"] = len(atoms)
        else:
            rec["natoms"] = sum(len(a) for a in built.values())

        rec["stage"] = "done"
        rec["ok"] = True
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {str(e)[:300]}"
        rec["trace"] = traceback.format_exc()[-600:]
    rec["seconds"] = round(time.time() - t0, 1)
    return rec


STAGES = ["parse", "spec", "gate1", "build", "checks", "assemble", "done"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default="data/out/eval")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = PROMPTS[: args.limit] if args.limit else PROMPTS

    results = []
    with (out / "results.jsonl").open("w") as fh:
        for i, (cat, prompt) in enumerate(prompts):
            rec = run_one(cat, prompt)
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            flag = "OK " if rec["ok"] else f"FAIL@{rec['stage']}"
            print(f"[{i + 1:3d}/{len(prompts)}] {flag:14} {rec['seconds']:6.1f}s "
                  f"{cat:14} {prompt[:60]}", flush=True)

    summary = {"n": len(results),
               "ok": sum(r["ok"] for r in results),
               "by_category": {}, "by_stage_failed": {}}
    for r in results:
        c = summary["by_category"].setdefault(
            r["category"], {"n": 0, "ok": 0, "failures": []})
        c["n"] += 1
        c["ok"] += int(r["ok"])
        if not r["ok"]:
            c["failures"].append({"prompt": r["prompt"], "stage": r["stage"],
                                  "error": r["error"]})
            summary["by_stage_failed"][r["stage"]] = \
                summary["by_stage_failed"].get(r["stage"], 0) + 1
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n{summary['ok']}/{summary['n']} prompts fully OK — "
          f"summary at {out / 'summary.json'}")


if __name__ == "__main__":
    main()
