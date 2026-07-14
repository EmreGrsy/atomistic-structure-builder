"""Generate docs/index.html — the documentation page with live 3D showcases.

Builds the six showcase structures with the real pipeline functions, embeds
them into a self-contained page (3Dmol.js from CDN), and — when an evaluation
summary exists at data/out/eval/summary.json — renders the benchmark section.

    python scripts/make_docs_page.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

APP_NAME = "Atomistic Structure Builder"
APP_VERSION = "0.1.0"

# locked house style (see mtagent/viewer.py)
BG, TEXT, CELLC, OUTLINE = "#efe9e1", "#5a4c40", "#a3927f", 0.10
STICK, VDW_FACTOR, OPACITY = 0.12, 0.44, 0.5


def _xyz(atoms, label):
    return f"{len(atoms)}\n{label}\n" + "\n".join(
        f"{s} {p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"
        for s, p in zip(atoms.get_chemical_symbols(), atoms.positions))


def _edges(atoms_list):
    allpos = np.vstack([a.positions for a in atoms_list])
    src = max(atoms_list, key=lambda a: a.cell.volume)
    if src.cell.volume > 1e-6:
        cell = np.array(src.cell)
        origin = allpos.mean(axis=0) - cell.sum(axis=0) / 2
    else:
        ext = allpos.max(axis=0) - allpos.min(axis=0) + 10.0
        cell, origin = np.diag(ext), allpos.min(axis=0) - 5.0
    corners = [origin + v for v in (0*cell[0], cell[0], cell[1], cell[2],
               cell[0]+cell[1], cell[0]+cell[2], cell[1]+cell[2], cell.sum(axis=0))]
    pairs = [(0,1),(0,2),(0,3),(1,4),(1,5),(2,4),(2,6),(3,5),(3,6),(4,7),(5,7),(6,7)]
    return [[corners[i].round(3).tolist(), corners[j].round(3).tolist()]
            for i, j in pairs]


def build_showcases() -> list[dict]:
    from mtagent import assemble
    from mtagent.cluster import build_cluster
    from mtagent.nanostructures import (build_magnetite_slab, build_nanotube,
                                        build_rutile_slab, build_sheet)
    from mtagent.pubchem import get_molecule
    from mtagent.solvent import build_solvent_box, solvate
    from mtagent.wulff import build_magnetite_wulff

    water = get_molecule("water")
    ethanol = get_molecule("ethanol")
    cases = []

    np_ = build_magnetite_wulff(diameter=26.0)
    solv = solvate(np_, build_solvent_box(water, box_size=40.0))
    n_np = solv.info["solvation"]["solute_atoms"]
    cases.append(dict(
        title="Solvated nanoparticle",
        prompt="a 2.2 nm magnetite nanoparticle in water",
        caption="Wulff-constructed Fe3O4 particle (literature facet energies), "
                "carve-and-insert solvation; the solvent renders translucent.",
        solid=solv[:n_np], trans=solv[n_np:], cellsrc=solv))

    cnt = build_nanotube(10, 10, length=12)
    filled = assemble.fill_inside(cnt, ethanol, n=12)
    cases.append(dict(
        title="Confined liquid",
        prompt="12 ethanol molecules inside a (10,10) carbon nanotube",
        caption="packmol cylinder fill; the enclosing tube wall renders "
                "translucent so the guest phase stays visible.",
        solid=filled[len(cnt):], trans=filled[:len(cnt)], cellsrc=filled))

    tio2 = build_rutile_slab((1, 1, 0), thickness=8.0, width=25.0, vacuum=14.0)
    coated = assemble.coat(tio2, water, n=80)
    cases.append(dict(
        title="Solid–liquid interface",
        prompt="water on a rutile TiO2 (110) surface",
        caption="The slab's vacuum is filled at liquid density inside one "
                "periodic cell; interfaces render fully solid.",
        solid=coated, trans=None, cellsrc=coated))

    sheet = build_sheet("graphene", width=24, vacuum=10)
    gsand = assemble.sandwich(sheet, water, n=80)
    cases.append(dict(
        title="2D confinement",
        prompt="80 water molecules between two graphene sheets",
        caption="A water film packed between the sheets with symmetric "
                "wall clearances; one periodic cell.",
        solid=gsand, trans=None, cellsrc=gsand))

    np20 = build_magnetite_wulff(diameter=24.0)
    sc = build_cluster(np20, n=4, gap=10.0, lattice="fcc")
    cases.append(dict(
        title="Nanoparticle supercrystal",
        prompt="an FCC supercrystal of 4 magnetite nanoparticles",
        caption="ASE builds the particle once; Moltemplate instantiates it "
                "on fcc superlattice sites (KG-validated .lt).",
        solid=sc, trans=None, cellsrc=sc))

    mag = build_magnetite_slab((0, 0, 1), thickness=8.0, vacuum=12.0, nx=3, ny=3)
    rut = build_rutile_slab((1, 1, 0), thickness=8.0, vacuum=12.0, nx=4, ny=9)
    hs = assemble.sandwich(mag, water, top=rut)
    cases.append(dict(
        title="Hetero-interface sandwich",
        prompt="water between a magnetite 001 slab and a rutile 110 slab",
        caption="Supercells are lattice-matched automatically; the top slab "
                "is strained epitaxially (recorded, 12% cap).",
        solid=hs, trans=None, cellsrc=hs))
    return cases


def sphere_radii(cases) -> dict:
    from mtagent.viewer import sphere_radius
    els = set()
    for c in cases:
        for a in (c["solid"], c["trans"]):
            if a is not None:
                els |= set(a.get_chemical_symbols())
    return {el: sphere_radius(el) for el in sorted(els)}


def metrics_section() -> str:
    path = ROOT / "data/out/eval/summary.json"
    if not path.exists():
        return ""
    s = json.loads(path.read_text())
    cats = sorted(s["by_category"].items())
    # single-hue horizontal bars: magnitude only (validated palette, blue-700)
    bar, track, ink = "#2c5aa0", "rgba(90,76,64,.12)", TEXT
    rows = []
    for name, d in cats:
        pct = 100.0 * d["ok"] / d["n"]
        rows.append(
            f'<div class="brow"><div class="blab">{name.replace("_", " ")}</div>'
            f'<div class="btrack"><div class="bfill" style="width:{pct:.0f}%"></div>'
            f'</div><div class="bval">{d["ok"]}/{d["n"]}</div></div>')
    total = f'{s["ok"]}/{s["n"]}'
    return f"""
  <h2>Benchmark</h2>
  <p class="doc">A {s['n']}-prompt evaluation spanning ten task categories, each
  prompt executed through the complete pipeline (parse &rarr; specification
  &rarr; static validation &rarr; sandboxed build &rarr; geometry checks &rarr;
  assembly). A prompt counts as successful only if every stage passes.
  Overall: <b>{total}</b> prompts fully successful.</p>
  <div class="bars" role="img" aria-label="success rate per category">
    {''.join(rows)}
  </div>
  <style>
    .brow {{ display:flex; align-items:center; gap:10px; margin:4px 0; }}
    .blab {{ width:170px; font-size:13px; color:{ink}; text-align:right; }}
    .btrack {{ flex:1; height:14px; background:{track}; border-radius:4px; }}
    .bfill {{ height:100%; background:{bar}; border-radius:4px; }}
    .bval {{ width:48px; font-size:12.5px; color:{ink}; opacity:.75; }}
  </style>"""


def main() -> None:
    cases = build_showcases()
    radii = sphere_radii(cases)

    data, cards = {}, []
    for i, c in enumerate(cases):
        parts = [c["solid"]] + ([c["trans"]] if c["trans"] is not None else [])
        data[f"s{i}"] = {
            "solid": _xyz(c["solid"], c["title"]),
            "trans": _xyz(c["trans"], c["title"]) if c["trans"] is not None else None,
            "edges": _edges([c["cellsrc"]]),
        }
        natoms = sum(len(p) for p in parts)
        cards.append(f"""
      <div class="card">
        <div class="cardhead">
          <div class="cardtitle">{c['title']}</div>
          <div class="prompt">&ldquo;{c['prompt']}&rdquo;</div>
        </div>
        <div class="viewer" id="v{i}"></div>
        <div class="caption">{c['caption']} <span class="n">{natoms} atoms.</span></div>
      </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{APP_NAME} — documentation</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; background:{BG}; color:{TEXT};
         font-family:'Segoe UI', system-ui, sans-serif; }}
  main {{ max-width:1200px; margin:0 auto; padding:36px 24px 64px; }}
  h1 {{ font-size:28px; margin:0; }}
  .tagline {{ font-size:15px; opacity:.8; margin:4px 0 0; }}
  .version {{ font-size:12.5px; opacity:.55; }}
  h2 {{ font-size:19px; margin:36px 0 8px; border-bottom:1px solid {CELLC};
        padding-bottom:4px; }}
  p.doc, li {{ font-size:14.5px; line-height:1.55; }}
  ol {{ padding-left:22px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
           gap:18px; margin-top:14px; }}
  .card {{ background:#f5f0e8; border:1px solid {CELLC}44; border-radius:12px;
           overflow:hidden; }}
  .cardhead {{ padding:12px 14px 8px; }}
  .cardtitle {{ font-weight:700; font-size:15px; }}
  .prompt {{ font-size:12.5px; margin-top:3px; font-family:ui-monospace,monospace;
             background:{BG}; border:1px solid {CELLC}55; border-radius:6px;
             padding:4px 8px; display:inline-block; }}
  .viewer {{ width:100%; height:380px; position:relative; }}
  .caption {{ padding:8px 14px 12px; font-size:12.5px; opacity:.8; }}
  .n {{ opacity:.7; }}
  code {{ background:#f5f0e8; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<main>
  <h1>{APP_NAME}</h1>
  <p class="tagline">A knowledge-graph-grounded atomistic structure builder —
  the code you see is the code that runs. Geometry only.</p>
  <p class="version">v{APP_VERSION}</p>

  <h2>Overview</h2>
  <p class="doc">Natural-language requests are converted into validated,
  atom-resolved 3D structures: nanoparticles, surfaces, interfaces, confined
  liquids, and nanoparticle superlattices. The deliverable of every build is
  the geometry itself — an interactive view and an <code>.xyz</code> file
  carrying the full simulation cell. Structures are geometric: crystal
  truncations are not reconstructed and assemblies are not equilibrated.</p>

  <h2>Pipeline</h2>
  <ol>
    <li><b>Parse</b> — the request is decomposed into typed constituents
      (nanoparticle, surface slab, bulk crystal, molecule, solvent box,
      nanotube) and relations (<i>inside, around, coated_by, on, between</i>).</li>
    <li><b>Retrieve</b> — real function signatures and constraints are pulled
      from two knowledge graphs: one introspected from the installed ASE
      (2,392 entries), one extracted from the Moltemplate manual. No code is
      generated without this evidence.</li>
    <li><b>Clarify</b> — only genuinely missing parameters are asked;
      everything else takes registry defaults.</li>
    <li><b>Propose</b> — a build snippet is written per constituent with the
      retrieved evidence in-prompt. The snippet shown is the code executed.</li>
    <li><b>Validate &amp; build</b> — three gates: static validation against
      the knowledge graphs, sandboxed execution, and geometric verification
      (finite coordinates, no unphysical contacts).</li>
    <li><b>Assemble</b> — constituents are combined per the stated relations:
      packmol for liquids, Moltemplate for repeated units (ligand shells,
      superlattices).</li>
  </ol>

  <h2>Toolchain</h2>
  <p class="doc">ASE builds the unit &middot; packmol packs liquids &middot;
  Moltemplate assembles repeated structures &middot; PubChem supplies
  molecular coordinates &middot; an LLM parses language and drafts snippets
  &middot; 3Dmol.js renders. Any element (conventional cells), 16 compound
  crystals, 2D sheets (graphene, h-BN), any Miller termination including
  4-index hexagonal notation, N&times;M supercells, and hetero-interfaces
  with automatic lattice matching.</p>

  <h2>Examples</h2>
  <p class="doc">Each example below is a real build produced by the prompt
  shown, rendered live — drag to rotate, scroll to zoom. Enclosing phases
  (solvent shells, tube walls) render at 50% opacity; interfaces stay solid.
  Sphere radii are proportional to van der Waals radii.</p>
  <div class="grid">{''.join(cards)}</div>
  {metrics_section()}
</main>
<script>
  const DATA = {json.dumps(data)};
  const RADII = {json.dumps(radii)};
  function makeViewer(id, d) {{
    const v = $3Dmol.createViewer(id, {{backgroundColor: "{BG}", orthographic: true}});
    v.addModel(d.solid, "xyz");
    if (d.trans) v.addModel(d.trans, "xyz");
    v.setViewStyle({{style: "outline", color: "black", width: {OUTLINE}}});
    for (const [el, r] of Object.entries(RADII)) {{
      v.setStyle({{model: 0, elem: el}}, {{stick: {{radius: {STICK}, colorscheme: "Jmol"}},
        sphere: {{radius: r, colorscheme: "Jmol"}}}});
      if (d.trans) v.setStyle({{model: 1, elem: el}},
        {{stick: {{radius: {STICK}, colorscheme: "Jmol", opacity: {OPACITY}}},
          sphere: {{radius: r, colorscheme: "Jmol", opacity: {OPACITY}}}}});
    }}
    for (const e of d.edges) v.addLine({{start: {{x: e[0][0], y: e[0][1], z: e[0][2]}},
      end: {{x: e[1][0], y: e[1][1], z: e[1][2]}}, color: "{CELLC}"}});
    v.zoomTo(); v.zoom(1.1); v.render();
  }}
  const PENDING = Object.keys(DATA).map((k, i) => ["v" + i, k]);
  const obs = new IntersectionObserver(es => {{
    for (const en of es) {{
      const hit = PENDING.find(p => p[0] === en.target.id);
      if (en.isIntersecting && hit) {{
        makeViewer(hit[0], DATA[hit[1]]);
        PENDING.splice(PENDING.indexOf(hit), 1);
        obs.unobserve(en.target);
      }}
    }}
  }}, {{rootMargin: "200px"}});
  document.querySelectorAll(".viewer").forEach(el => obs.observe(el));
</script>
</body>
</html>"""

    out = ROOT / "docs" / "index.html"
    out.parent.mkdir(exist_ok=True)
    out.write_text(html)
    print(f"wrote {out} ({len(html)} bytes, {len(cases)} showcases, "
          f"metrics={'yes' if metrics_section() else 'no'})")


if __name__ == "__main__":
    main()
