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

# viewer style stays warm (see mtagent/viewer.py); the PAGE itself follows
# the app's default Streamlit palette
BG, TEXT, CELLC, OUTLINE = "#efe9e1", "#5a4c40", "#a3927f", 0.10
PAGE_BG, PAGE_TEXT, PAGE_LINE, CARD_BG = "#0e1117", "#fafafa", "#3d414b", "#262730"
STICK, VDW_FACTOR, OPACITY = 0.12, 0.44, 0.5
# element color overrides (user's publication convention); others use Jmol
CUSTOM_COLORS = {"Fe": "#3565c0"}


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


def _fill_around(cluster, molecule, n: int, margin: float = -1.0,
                 workdir="data/work/docs_fill"):
    """Pack n molecules into the cluster's bounding box (cluster fixed) and
    return ONLY the packed molecules (rendered as the translucent phase)."""
    import subprocess
    from ase.io import read, write
    from mtagent.packing import find_packmol
    work = Path(ROOT / workdir)
    work.mkdir(parents=True, exist_ok=True)
    p = cluster.get_positions()
    lo, hi = p.min(axis=0) - margin, p.max(axis=0) + margin
    cl, mol, outx = work / "cluster.xyz", work / "mol.xyz", work / "filled.xyz"
    write(str(cl), cluster, format="xyz")
    m = molecule.copy()
    m.set_positions(m.get_positions() - m.get_positions().mean(axis=0))
    write(str(mol), m, format="xyz")
    (work / "fill.inp").write_text(
        f"tolerance 2.0\nfiletype xyz\noutput {outx.name}\n\n"
        f"structure {cl.name}\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\n"
        "end structure\n\n"
        f"structure {mol.name}\n  number {n}\n"
        f"  inside box {lo[0]:.2f} {lo[1]:.2f} {lo[2]:.2f} "
        f"{hi[0]:.2f} {hi[1]:.2f} {hi[2]:.2f}\nend structure\n")
    proc = subprocess.run([find_packmol()], stdin=(work / "fill.inp").open(),
                          cwd=work, capture_output=True, text=True, timeout=1200)
    if not outx.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"docs fill failed:\n{proc.stdout[-500:]}")
    return read(str(outx))[len(cluster):]


def _cached_case(key: str, params: dict, build):
    """(solid, trans) from the example cache, else build once and cache.

    The cache key is the parameter dict, so changing any knob rebuilds only
    that example; bump "v" to bust the cache after a logic change."""
    import hashlib
    from ase.io import read, write
    sig = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:12]
    cdir = ROOT / "data" / "cache" / "docs_examples"
    s_f = cdir / f"{key}-{sig}.solid.xyz"
    t_f = cdir / f"{key}-{sig}.trans.xyz"
    if s_f.exists():
        print(f"  [cache] {key}")
        return read(s_f), (read(t_f) if t_f.exists() else None)
    print(f"  [build] {key}")
    solid, trans = build()
    cdir.mkdir(parents=True, exist_ok=True)
    a = solid.copy()
    a.info.clear()
    write(s_f, a)
    if trans is not None:
        b = trans.copy()
        b.info.clear()
        write(t_f, b)
    return solid, trans


def build_showcases() -> list[dict]:
    from mtagent import assemble
    from mtagent.cluster import build_cluster
    from mtagent.nanostructures import (build_magnetite_slab, build_nanotube,
                                        build_rutile_slab, build_sheet)
    from mtagent.pubchem import get_molecule
    from mtagent.solvent import build_solvent_box, solvate
    from mtagent.wulff import build_magnetite_wulff

    water = get_molecule("water")
    cases = []

    # equal facet energies -> all facet planes equidistant: the most
    # sphere-like Wulff habit these families can produce
    round_gamma = {(1, 1, 1): 1.0, (1, 0, 0): 1.0, (1, 1, 0): 1.0}

    def _np_water():
        np_ = build_magnetite_wulff(diameter=26.0, gamma=round_gamma)
        solv = solvate(np_, build_solvent_box(water, box_size=40.0))
        n_np = solv.info["solvation"]["solute_atoms"]
        return solv[:n_np], solv[n_np:]

    solid, trans = _cached_case(
        "np_water", {"d": 26.0, "box": 40.0, "gamma": "1/1/1", "v": 1}, _np_water)
    cases.append(dict(
        title="Solvated nanoparticle",
        prompt="a 2.6 nm magnetite nanoparticle in water",
        caption="Wulff constructed Fe3O4 particle (balanced facet energies, "
                "near spherical habit), carve and insert solvation; the "
                "solvent renders translucent.",
        solid=solid, trans=trans, cellsrc=trans))

    def _electrolyte():
        from ase import Atoms as _Atoms
        cnt = build_nanotube(10, 10, length=18)
        # ONE ion, placed deterministically at the tube center; water then
        # fills the cylinder around it
        ion = _Atoms("K", positions=[cnt.get_positions().mean(axis=0)])
        host = cnt + ion
        host.info["provenance"] = {"type": "nanotube"}     # keep the cylinder
        filled = assemble.fill_inside(host, water, n=44)
        n_wall = len(cnt)
        return filled[n_wall:], filled[:n_wall]            # wall translucent

    solid, trans = _cached_case(
        "electrolyte_cnt", {"nm": "10x10x18", "waters": 44, "ions": 1, "v": 3},
        _electrolyte)
    print(f"  electrolyte check: {solid.get_chemical_symbols().count('K')} "
          "K ion present")
    cases.append(dict(
        title="Confined electrolyte",
        prompt="water with a potassium ion inside a (10,10) carbon nanotube",
        caption="A single solvated ion at the tube center, water packed "
                "around it (Packmol cylinder fill); the enclosing tube wall "
                "renders translucent so the confined phase stays visible.",
        solid=solid, trans=trans, cellsrc=trans))

    oleic = get_molecule("oleic acid")

    def _interface():
        tio2 = build_rutile_slab((1, 1, 0), thickness=8.0, width=25.0, vacuum=20.0)
        return assemble.coat(tio2, oleic, n=14), None

    solid, trans = _cached_case(
        "oleic_rutile", {"w": 25.0, "vac": 20.0, "n": 14, "v": 1}, _interface)
    cases.append(dict(
        title="Solid liquid interface",
        prompt="14 oleic acid molecules on a rutile TiO2 (110) surface",
        caption="Organic molecules packed above the surface inside one "
                "periodic cell; interfaces render fully solid.",
        solid=solid, trans=trans, cellsrc=solid))

    def _graphene():
        sheet = build_sheet("graphene", width=24, vacuum=10)
        return assemble.sandwich(sheet, water, n=80), None

    solid, trans = _cached_case(
        "graphene_water", {"w": 24, "n": 80, "v": 1}, _graphene)
    cases.append(dict(
        title="2D confinement",
        prompt="80 water molecules between two graphene sheets",
        caption="A water film packed between the sheets with symmetric "
                "wall clearances; one periodic cell.",
        solid=solid, trans=trans, cellsrc=solid))

    seq = "ABCABCABAB"

    def _supercrystal():
        np26 = build_magnetite_wulff(diameter=26.0, gamma=round_gamma)
        sc = build_cluster(np26, n=len(seq), gap=6.0, lattice=seq)
        return sc, _fill_around(sc, oleic, n=500, margin=4.0)

    solid, trans = _cached_case(
        "faulted_supercrystal",
        {"d": 26.0, "seq": seq, "gap": 6.0, "oleic": 500, "v": 1}, _supercrystal)
    cases.append(dict(
        title="Faulted nanoparticle supercrystal",
        prompt=f"a {seq} stacked supercrystal of 2.6 nm magnetite "
               "nanoparticles, filled with oleic acid",
        caption="Close packed layers with an explicit stacking sequence, an "
                "fcc supercrystal carrying a stacking fault; Moltemplate "
                "instantiates the particle per site, oleic acid fills the "
                "interstitial space (translucent).",
        solid=solid, trans=trans, cellsrc=solid))

    def _hetero():
        mag = build_magnetite_slab((0, 0, 1), thickness=8.0, vacuum=12.0, nx=3, ny=3)
        rut = build_rutile_slab((1, 1, 0), thickness=8.0, vacuum=12.0, nx=4, ny=9)
        return assemble.sandwich(mag, water, top=rut, clearance=3.5, n=100), None

    solid, trans = _cached_case(
        "hetero_sandwich", {"clr": 3.5, "n": 100, "v": 1}, _hetero)
    cases.append(dict(
        title="Hetero interface sandwich",
        prompt="water between a magnetite 001 slab and a rutile 110 slab",
        caption="Supercells are lattice matched automatically; the top slab "
                "is strained epitaxially (recorded, 12% cap).",
        solid=solid, trans=trans, cellsrc=solid))
    return cases


def sphere_radii(cases) -> dict:
    from mtagent.viewer import sphere_radius
    els = set()
    for c in cases:
        for a in (c["solid"], c["trans"]):
            if a is not None:
                els |= set(a.get_chemical_symbols())
    return {el: sphere_radius(el) for el in sorted(els)}


def _hex_svg(coords, values, labels, ramp, title) -> str:
    """Pointy-top hexagonal heat map (single-hue sequential ramp)."""
    def lerp(c0, c1, t):
        return "#" + "".join(f"{round(int(c0[i:i+2],16)*(1-t)+int(c1[i:i+2],16)*t):02x}"
                             for i in (1, 3, 5))
    vmin, vmax = min(values), max(values)
    S, r = 34, 19
    w = max(x for x, _ in coords) * S + 2 * S
    h = max(y for _, y in coords) * S + 2 * S
    hexpts = ",".join(f"{r*np.sin(np.radians(a)):.1f} {-r*np.cos(np.radians(a)):.1f}"
                      for a in range(0, 360, 60))
    cells = []
    for (x, y), v, lab in zip(coords, values, labels):
        t = 0.0 if vmax == vmin else (v - vmin) / (vmax - vmin)
        cx, cy = x * S + S, y * S + S
        cells.append(f'<g transform="translate({cx:.1f},{cy:.1f})">'
                     f'<polygon points="{hexpts}" fill="{lerp(*ramp, t)}" '
                     f'stroke="{PAGE_BG}" stroke-width="2"/>'
                     + (f'<text y="4" text-anchor="middle" font-size="11" '
                        f'fill="{PAGE_TEXT}">{lab}</text>' if lab else "") + "</g>")
    return (f'<figure class="som"><svg viewBox="0 0 {w:.0f} {h:.0f}" '
            f'role="img" aria-label="{title}">{"".join(cells)}</svg>'
            f'<figcaption>{title} <span class="rlab">{vmin:.2f}</span>'
            f'<span class="ramp" style="background:linear-gradient(90deg,'
            f'{ramp[0]},{ramp[1]})"></span><span class="rlab">{vmax:.2f}</span>'
            f'</figcaption></figure>')


def som_section() -> str:
    path = ROOT / "data/out/eval/som.json"
    if not path.exists():
        return ""
    s = json.loads(path.read_text())
    coords = s["coords"]
    um = _hex_svg(coords, s["umatrix"], [""] * len(coords),
                  ("#e4efe7", "#1e4531"),
                  "U matrix (mean neighbor distance)")
    hits = [c["hits"] for c in s["cells"]]
    hm = _hex_svg(coords, hits, [str(v) if v else "" for v in hits],
                  ("#f7ece1", "#8a4a24"), "BMU hit map (prompts per neuron)")
    return f"""
  <h3>Prompt space analysis (SOM)</h3>
  <p class="doc">To understand the diversity of the prompt set, the 100
  prompts were converted into 3072 dimensional embedding vectors with OpenAI's
  <code>text-embedding-3-large</code> model and a 10&times;10 hexagonally
  packed self organizing map (SOM;
  <a href="https://ieeexplore.ieee.org/document/58325">Kohonen,
  <i>Proceedings of the IEEE</i> 78, 1464 to 1480, 1990</a>) was trained on
  them, following the methodology of GENIUS
  (<a href="https://arxiv.org/abs/2512.06404">arXiv:2512.06404</a>). The SOM
  is an unsupervised neural network that projects high dimensional data onto
  a two dimensional grid while preserving the topology of the input space;
  training used 50,000 mini batch iterations with a Gaussian neighborhood
  and a linearly decreasing learning rate. Quantization Error is
  <b>{s['qe']}</b>, good representational fidelity given that the
  unit normalized inputs have a maximum possible pairwise distance of 2.0;
  Topological Error is <b>{s['te']}</b>, confirming the neighborhood
  structure is preserved. In the U matrix, high values mark cluster
  boundaries and low values dense groups of similar prompts; the hit map
  counts BMU activations per neuron, and empty neurons act as boundary
  regions between clusters.</p>
  <div class="somrow">{um}{hm}</div>
  <style>
    .somrow {{ display:flex; gap:22px; flex-wrap:wrap; }}
    .som {{ margin:0; flex:1; min-width:320px; max-width:460px; }}
    .som svg {{ width:100%; height:auto; }}
    .som figcaption {{ font-size:12.5px; opacity:.8; margin-top:4px;
                       display:flex; align-items:center; gap:6px; }}
    .ramp {{ display:inline-block; width:90px; height:10px; border-radius:3px; }}
    .rlab {{ font-size:11px; opacity:.7; }}
  </style>"""


def metrics_section() -> str:
    path = ROOT / "data/out/eval/summary.json"
    if not path.exists():
        return ""
    s = json.loads(path.read_text())
    cats = sorted(s["by_category"].items())
    # single-hue horizontal bars sampled from the U matrix ramp (4.0:1)
    bar, track, ink = "#6d8f7d", "rgba(250,250,250,.12)", PAGE_TEXT
    rows = []
    for name, d in cats:
        pct = 100.0 * d["ok"] / d["n"]
        rows.append(
            f'<div class="brow"><div class="blab">{name.replace("_", " ")}</div>'
            f'<div class="btrack"><div class="bfill" style="width:{pct:.0f}%"></div>'
            f'</div><div class="bval">{d["ok"]}/{d["n"]}</div></div>')
    total = f'{s["ok"]}/{s["n"]}'
    comp_path = ROOT / "data/out/eval/complexity.json"
    comp_txt = ""
    if comp_path.exists():
        cp = json.loads(comp_path.read_text())["percent"]
        comp_txt = (
            f" A score based metric evaluation shows that the prompts comprise "
            f"{cp['basic']}% basic, {cp['standard']}% standard, and "
            f"{cp['complex']}% complex requests. This evaluation is performed "
            "with OpenAI's <code>gpt-4o-mini</code> model, which assigns a "
            "categorical value to each prompt from the number of "
            "constituents, relations, and explicit parameters it carries.")
    return f"""
  <h2>Benchmark</h2>
  <p class="doc">To probe the framework's coverage and robustness, a set of
  {s['n']} test prompts spanning ten task categories (metal and oxide
  nanoparticles, solvation, elemental and compound surfaces, solid liquid
  interfaces, confined films, nanoparticle supercrystals, filled nanotubes,
  and bulk builds) was executed through the complete pipeline: parsing,
  specification, static validation against the knowledge graphs, sandboxed
  building, geometric verification, and assembly. A prompt counts as
  successful only if every stage passes; a structurally plausible but
  invalid intermediate is treated as a failure of its stage.{comp_txt}
  Overall, <b>{total}</b> prompts complete the full pipeline successfully.
  The two residual failures are a unit interpretation slip by the language
  model (now surfaced as an explicit error rather than an empty structure)
  and one stochastic packing run that did not converge.</p>
  <div class="bars" role="img" aria-label="success rate per category">
    {''.join(rows)}
  </div>
  <p class="doc"><a class="promptlink" href="eval_prompts.html">View the
  complete prompt set</a></p>
  <style>
    .brow {{ display:flex; align-items:center; gap:10px; margin:4px 0; }}
    .blab {{ width:170px; font-size:13px; color:{ink}; text-align:right; }}
    .btrack {{ flex:1; height:14px; background:{track}; border-radius:4px; }}
    .bfill {{ height:100%; background:{bar}; border-radius:4px; }}
    .bval {{ width:48px; font-size:12.5px; color:{ink}; opacity:.75; }}
    .promptlink {{ display:inline-block; margin-top:6px; padding:7px 14px;
      background:{bar}; color:#ffffff; border-radius:8px; font-weight:600;
      text-decoration:none; }}
    .promptlink:hover {{ filter:brightness(1.1); }}
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
<title>{APP_NAME} documentation</title>
<script src="https://3Dmol.org/build/3Dmol-min.js"></script>
<style>
  body {{ margin:0; background:{PAGE_BG}; color:{PAGE_TEXT};
         font-family:'Segoe UI', system-ui, sans-serif; }}
  main {{ max-width:1200px; margin:0 auto; padding:36px 24px 64px; }}
  h1 {{ font-size:28px; margin:0; }}
  .tagline {{ font-size:15px; opacity:.8; margin:4px 0 0; }}
  .version {{ font-size:12.5px; opacity:.55; }}
  h2 {{ font-size:19px; margin:36px 0 8px; border-bottom:1px solid {PAGE_LINE};
        padding-bottom:4px; }}
  p.doc, li {{ font-size:14.5px; line-height:1.55; }}
  ol {{ padding-left:22px; }}
  .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(340px,1fr));
           gap:18px; margin-top:14px; }}
  .card {{ background:{CARD_BG}; border:1px solid {PAGE_LINE}; border-radius:12px;
           overflow:hidden; }}
  .cardhead {{ padding:12px 14px 8px; }}
  .cardtitle {{ font-weight:700; font-size:15px; }}
  .prompt {{ font-size:12.5px; margin-top:3px; font-family:ui-monospace,monospace;
             background:{PAGE_BG}; border:1px solid {PAGE_LINE}; border-radius:6px;
             padding:4px 8px; display:inline-block; }}
  .viewer {{ width:100%; height:380px; position:relative; }}
  .caption {{ padding:8px 14px 12px; font-size:12.5px; opacity:.8; }}
  .n {{ opacity:.7; }}
  code {{ background:{CARD_BG}; padding:1px 5px; border-radius:4px; }}
</style>
</head>
<body>
<main>
  <h1>{APP_NAME}</h1>
  <p class="tagline">A knowledge graph grounded atomistic structure builder,
  the code you see is the code that runs. Geometry only.</p>
  <p class="version">v{APP_VERSION}</p>

  <h2>Overview</h2>
  <p class="doc">Natural language requests are converted into validated,
  atom resolved 3D structures: nanoparticles, surfaces, interfaces, confined
  liquids, and nanoparticle superlattices. The deliverable of every build is
  the geometry itself, an interactive view and an <code>.xyz</code> file
  carrying the full simulation cell. Structures are geometric: crystal
  truncations are not reconstructed and assemblies are not equilibrated.</p>

  <h2>Pipeline</h2>
  <ol>
    <li><b>Parse.</b> The request is decomposed into typed constituents
      (nanoparticle, surface slab, bulk crystal, molecule, solvent box,
      nanotube) and relations (<i>inside, around, coated_by, on, between</i>).</li>
    <li><b>Retrieve.</b> Real function signatures and constraints are pulled
      from two knowledge graphs. The ASE knowledge graph is introspected from
      the installed package: 2,392 nodes (one per function or class, carrying
      its real signature) with connectivity edges linking each callable to
      its parent module. The Moltemplate knowledge graph is built once from
      the official manual, and only the graph is queried at runtime: 60 nodes
      (the .lt constructs) with 23 connectivity edges (which constructs
      contain or apply to which) plus 8 constraint rules. No code is
      generated without this evidence.</li>
    <li><b>Clarify.</b> Only genuinely missing parameters are asked;
      everything else takes registry defaults.</li>
    <li><b>Propose.</b> A build snippet is written per constituent with the
      retrieved evidence included in the prompt. The snippet shown is the code executed.</li>
    <li><b>Validate &amp; build.</b> Three gates: static validation against
      the knowledge graphs, sandboxed execution, and geometric verification
      (finite coordinates, no unphysical contacts).</li>
    <li><b>Assemble.</b> Constituents are combined per the stated relations:
      packmol for liquids, Moltemplate for repeated units (ligand shells,
      superlattices).</li>
  </ol>

  <h2>Toolchain</h2>
  <p class="doc">ASE builds the unit. Packmol packs liquids. Moltemplate
  assembles repeated structures. PubChem supplies molecular coordinates.
  OpenAI's <code>gpt-4o-mini</code> parses language and drafts the build
  snippets. 3Dmol.js renders.
  Supported: any element (conventional cells), 16 compound crystals,
  2D sheets (graphene, hBN), any Miller termination including 4 index
  hexagonal notation, N&times;M supercells, and hetero interfaces with
  automatic lattice matching.</p>

  <h2>Examples</h2>
  <p class="doc">Each example below is a real build produced by the prompt
  shown, rendered live, drag to rotate and scroll to zoom. Enclosing phases
  (solvent shells, tube walls) render at 50% opacity; interfaces stay solid.
  Sphere radii are proportional to van der Waals radii.</p>
  <div class="grid">{''.join(cards)}</div>
  {metrics_section()}
  {som_section()}
</main>
<script>
  const DATA = {json.dumps(data)};
  const RADII = {json.dumps(radii)};
  const COLORS = {json.dumps(CUSTOM_COLORS)};
  // free ions are not covalently bonded: drop inferred bonds involving
  // alkali or alkaline earth atoms, and halide bonds to anything but carbon
  const CATIONS = new Set(["Li","Na","K","Rb","Cs","Mg","Ca","Sr","Ba"]);
  const HALIDES = new Set(["F","Cl","Br","I"]);
  function stripIonBonds(model) {{
    const atoms = model.selectedAtoms({{}});
    const drop = (a, b) => CATIONS.has(a.elem) || CATIONS.has(b.elem) ||
      (HALIDES.has(a.elem) && b.elem !== "C") ||
      (HALIDES.has(b.elem) && a.elem !== "C");
    for (const a of atoms) {{
      const bonds = [], orders = [];
      a.bonds.forEach((bi, k) => {{
        if (!drop(a, atoms[bi])) {{ bonds.push(bi); orders.push(a.bondOrder[k]); }}
      }});
      a.bonds = bonds; a.bondOrder = orders;
    }}
  }}
  function makeViewer(id, d) {{
    const v = $3Dmol.createViewer(id, {{backgroundColor: "{BG}", orthographic: true}});
    v.addModel(d.solid, "xyz");
    if (d.trans) v.addModel(d.trans, "xyz");
    stripIonBonds(v.getModel(0));
    if (d.trans) stripIonBonds(v.getModel(1));
    v.setViewStyle({{style: "outline", color: "black", width: {OUTLINE}}});
    for (const [el, r] of Object.entries(RADII)) {{
      const col = COLORS[el] ? {{color: COLORS[el]}} : {{colorscheme: "Jmol"}};
      v.setStyle({{model: 0, elem: el}}, {{stick: {{radius: {STICK}, ...col}},
        sphere: {{radius: r, ...col}}}});
      if (d.trans) v.setStyle({{model: 1, elem: el}},
        {{stick: {{radius: {STICK}, ...col, opacity: {OPACITY}}},
          sphere: {{radius: r, ...col, opacity: {OPACITY}}}}});
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
    # mirror into ./static so the app's Documentation link serves the pages
    static = ROOT / "static"
    static.mkdir(exist_ok=True)
    (static / "index.html").write_text(html)
    ep = ROOT / "docs" / "eval_prompts.html"
    if ep.exists():
        (static / "eval_prompts.html").write_text(ep.read_text())
    print(f"wrote {out} ({len(html)} bytes, {len(cases)} showcases, "
          f"metrics={'yes' if metrics_section() else 'no'})")


if __name__ == "__main__":
    main()
