"""LLM-scored prompt complexity classification (GENIUS methodology,
arXiv:2512.06404): each evaluation prompt is assigned basic, standard, or
complex by a language model. Writes data/out/eval/complexity.json.

    python scripts/prompt_complexity.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

_SYSTEM = """You classify user prompts for a molecular-structure builder by
complexity. Definitions:
- basic: one constituent, at most one explicit parameter (e.g. "a 3 nm gold
  nanoparticle", "bulk silicon").
- standard: one relation between constituents OR several explicit parameters
  (e.g. "30 water molecules on a gold 111 slab").
- complex: multiple relations, multiple constituents with coupled parameters,
  or unusual constructs (stacking sequences, hetero-interfaces, superlattices
  with packing choices).
Return JSON: {"label": "basic"|"standard"|"complex"} for the single prompt
given."""


def main() -> None:
    from mtagent.llm import chat_json
    results = [json.loads(ln) for ln in
               (ROOT / "data/out/eval/results.jsonl").read_text().splitlines()]
    prompts = [r["prompt"] for r in results]
    labels: list[str] = []
    for i, p in enumerate(prompts):                # one per call: batch outputs
        out = chat_json([{"role": "user", "content": p}], _SYSTEM)   # truncate
        lab = str(out.get("label", "")).lower()
        if lab not in ("basic", "standard", "complex"):
            lab = "standard"
        labels.append(lab)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(prompts)}")
    counts = {k: labels.count(k) for k in ("basic", "standard", "complex")}
    data = {"n": len(prompts), "counts": counts,
            "percent": {k: round(100.0 * v / len(prompts), 1)
                        for k, v in counts.items()},
            "labels": labels}
    (ROOT / "data/out/eval/complexity.json").write_text(json.dumps(data, indent=2))
    print(data["percent"])


if __name__ == "__main__":
    main()
