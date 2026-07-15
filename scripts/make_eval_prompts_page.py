"""Generate docs/eval_prompts.html from the evaluation results.

Reads data/out/eval/results.jsonl and data/out/eval/complexity.json and
writes the full prompt table in the documentation color scheme. Run after
scripts/eval_prompts.py (and scripts/prompt_complexity.py if the prompt set
changed). make_docs_page.py mirrors the file into static/.

    python scripts/make_eval_prompts_page.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE_BG, CARD_BG, TEXT, LINE = "#222834", "#151a22", "#e9edf4", "#39435a"

_HEAD = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Evaluation prompt set</title><style>
body {{ background:{PAGE_BG}; color:{TEXT}; font-family:'Segoe UI',system-ui,sans-serif;
       max-width:1050px; margin:0 auto; padding:32px 20px; }}
h1 {{ font-size:22px; }} p {{ font-size:14px; }}
a {{ color:#7f97b8; }}
table {{ border-collapse:collapse; width:100%; font-size:13px;
        background:{CARD_BG}; border:1px solid {LINE}; }}
th, td {{ text-align:left; padding:5px 9px; border-bottom:1px solid {LINE}; }}
th {{ border-bottom:2px solid #5c708f; }}
.ok {{ color:#9fb4d0; font-weight:600; }} .bad {{ color:#e08a7a; font-weight:600; }}
</style></head><body>
"""


def main() -> None:
    results = [json.loads(ln) for ln in
               (ROOT / "data/out/eval/results.jsonl").read_text().splitlines()]
    comp = json.loads((ROOT / "data/out/eval/complexity.json").read_text())
    labels, pct = comp["labels"], comp["percent"]
    ok = sum(r["ok"] for r in results)

    rows = ["<table><tr><th>#</th><th>category</th><th>complexity</th>"
            "<th>prompt</th>\n<th>outcome</th></tr>"]
    for i, r in enumerate(results):
        outcome = ('<td class=ok>pass</td>' if r["ok"]
                   else f'<td class=bad>fail at {r["stage"]}</td>')
        rows.append(
            f'<tr><td>{i + 1}</td><td>{r["category"].replace("_", " ")}</td>'
            f'<td>{labels[i]}</td><td>{r["prompt"]}</td>{outcome}</tr>')
    rows.append("</table>")

    html = (_HEAD
            + f"<h1>Evaluation prompt set, {len(results)} prompts</h1>\n"
            + f"<p><b>{ok}/{len(results)}</b> pass the full pipeline (parse, "
              "specification, static\nvalidation, sandboxed build, geometry "
              "checks, assembly). Complexity\ndistribution, scored with "
              f"gpt-4o-mini: basic {pct['basic']}%, "
              f"standard {pct['standard']}%, complex {pct['complex']}%.</p>\n"
            + "".join(rows) + "</body></html>")
    out = ROOT / "docs" / "eval_prompts.html"
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes, {ok}/{len(results)} pass)")


if __name__ == "__main__":
    main()
