"""Shared 3D geometry viewer.

Cheap and dependency-light: renders an .xyz structure with 3Dmol.js loaded from a
CDN into a self-contained HTML file. The same HTML string embeds directly into
Streamlit via ``st.components.v1.html(...)``.

House style (user-chosen, 2026-07-14): warm-gray page, toon ball-and-stick with
black ink outline, spheres proportional to real vdW radii, Jmol element colors,
corner axis gizmo with Miller-direction arrows. Components that ENCLOSE another
(solvation shell, filled-tube wall, NP ligand shell) render at 0.5 opacity;
interfaces (adsorbates/liquid layers on slabs) stay solid.

The alternate dark theme the user may switch to later is kept in ``THEMES``.
"""
from __future__ import annotations

import json
import os
import webbrowser
from pathlib import Path
from string import Template

THEMES = {
    "warm_gray":  dict(bg="#efe9e1", text="#5a4c40", cell="#a3927f", outline=0.10),
    "dark_slate": dict(bg="#20242c", text="#dfe3ea", cell="#58637a", outline=0.08),
}
THEME = THEMES["warm_gray"]

STICK_RADIUS = 0.12
VDW_FACTOR = 0.44            # toon ball-and-stick: spheres at 44% of the vdW radius
ENCLOSURE_OPACITY = 0.5
DEFAULT_AXES = [["[100]", [1, 0, 0]], ["[010]", [0, 1, 0]], ["[001]", [0, 0, 1]]]

_TEMPLATE = Template("""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>$title</title>
  <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
  <style>
    body { margin: 0; font-family: sans-serif; background: $bg; color: $text;
            display: flex; flex-direction: column; align-items: center; }
    #header { padding: 8px 12px 2px; font-size: 14px; font-weight: 600; }
    #legend { padding: 2px 12px 6px; font-size: 13px; display: flex;
               flex-wrap: wrap; gap: 12px; justify-content: center; }
    .chip { display: inline-flex; align-items: center; gap: 5px; }
    .dot { width: 13px; height: 13px; border-radius: 50%; display: inline-block;
            border: 1px solid rgba(0,0,0,.35); }
    #wrap { position: relative; }
    #viewer { width: ${width}px; height: ${height}px; position: relative; }
    #gizmo { position: absolute; left: 0; bottom: 0; pointer-events: none; }
  </style>
</head>
<body>
  <div id="header">$title &mdash; $natoms atoms</div>
  <div id="legend">$legend</div>
  <div id="wrap">
    <div id="viewer"></div>
    <canvas id="gizmo" width="104" height="104"></canvas>
  </div>
  <script>
    const XYZ_SOLID = $xyz_solid;
    const XYZ_TRANS = $xyz_trans;
    const EDGES = $edges;
    const AXES = $axes;
    const RADII = $radii;

    const viewer = $$3Dmol.createViewer("viewer",
        { backgroundColor: "$bg", orthographic: true });
    viewer.addModel(XYZ_SOLID, "xyz");
    if (XYZ_TRANS !== null) viewer.addModel(XYZ_TRANS, "xyz");
    viewer.setViewStyle({ style: "outline", color: "black", width: $outline });
    for (const [el, r] of Object.entries(RADII)) {
      viewer.setStyle({model: 0, elem: el},
        { stick: { radius: $stick, colorscheme: "Jmol" },
          sphere: { radius: r, colorscheme: "Jmol" } });
      if (XYZ_TRANS !== null)
        viewer.setStyle({model: 1, elem: el},
          { stick: { radius: $stick, colorscheme: "Jmol", opacity: $opacity },
            sphere: { radius: r, colorscheme: "Jmol", opacity: $opacity } });
    }
    for (const e of EDGES) {
      viewer.addLine({ start: {x: e[0][0], y: e[0][1], z: e[0][2]},
                       end:   {x: e[1][0], y: e[1][1], z: e[1][2]},
                       color: "$cellcolor" });
    }
    viewer.zoomTo();
    viewer.render();

    // ---- corner axis gizmo: arrows along crystallographic directions -------
    function rotv(q, v) {                 // rotate vector by quaternion [x,y,z,w]
      const [x, y, z, w] = q;
      const tx = 2*(y*v[2]-z*v[1]), ty = 2*(z*v[0]-x*v[2]), tz = 2*(x*v[1]-y*v[0]);
      return [v[0]+w*tx+(y*tz-z*ty), v[1]+w*ty+(z*tx-x*tz), v[2]+w*tz+(x*ty-y*tx)];
    }
    const gizmo = document.getElementById("gizmo");
    function drawGizmo(q) {
      const ctx = gizmo.getContext("2d");
      const c = 52, L = 26, AH = 6;
      ctx.clearRect(0, 0, 104, 104);
      const proj = AXES.map(([lab, v]) => [lab, rotv(q, v)])
                       .sort((p, r) => p[1][2] - r[1][2]);   // painter: back first
      for (const [lab, r] of proj) {
        const x = c + L*r[0], y = c - L*r[1];
        const len2d = Math.hypot(x - c, y - c);
        ctx.strokeStyle = "$text"; ctx.fillStyle = "$text";
        ctx.lineWidth = 2; ctx.lineCap = "round";
        ctx.globalAlpha = 0.45 + 0.55 * (r[2] + 1) / 2;      // depth cue
        if (len2d < 4) {                                     // points at the viewer
          ctx.beginPath(); ctx.arc(x, y, 3, 0, 7); ctx.fill();
        } else {
          const ux = (x - c) / len2d, uy = (y - c) / len2d;
          ctx.beginPath(); ctx.moveTo(c, c); ctx.lineTo(x, y); ctx.stroke();
          ctx.beginPath();                                   // arrowhead
          ctx.moveTo(x + AH*ux, y + AH*uy);
          ctx.lineTo(x - AH*uy*0.6, y + AH*ux*0.6);
          ctx.lineTo(x + AH*uy*0.6, y - AH*ux*0.6);
          ctx.closePath(); ctx.fill();
        }
        if (lab) {
          ctx.font = "bold 10px sans-serif"; ctx.textAlign = "center";
          ctx.fillText(lab, c + (L+16)*r[0], c - (L+16)*r[1] + 3);
        }
      }
      ctx.globalAlpha = 1;
    }
    viewer.setViewChangeCallback(view => drawGizmo(view.slice(4, 8)));
    drawGizmo(viewer.getView().slice(4, 8));
  </script>
</body>
</html>
""")


def sphere_radius(element: str) -> float:
    """Display sphere radius: real vdW radius (ASE/Bondi) x VDW_FACTOR.

    Elements missing from the Bondi table (most metals) fall back to
    covalent + 0.7 A, the standard approximation."""
    import numpy as np
    from ase.data import atomic_numbers, covalent_radii, vdw_radii
    z = atomic_numbers[element]
    r = vdw_radii[z]
    if not np.isfinite(r):
        r = covalent_radii[z] + 0.7
    return round(float(r) * VDW_FACTOR, 3)


def _legend_html(*xyzs: str) -> str:
    """Chips for every element present — same Jmol color code 3Dmol renders with."""
    from ase.data import atomic_numbers
    from ase.data.colors import jmol_colors
    counts: dict[str, int] = {}
    for xyz in xyzs:
        if not xyz:
            continue
        for line in xyz.strip().splitlines()[2:]:
            tok = line.split()
            if tok:
                counts[tok[0]] = counts.get(tok[0], 0) + 1
    chips = []
    for el, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        z = atomic_numbers.get(el)
        r, g, b = ((jmol_colors[z] * 255).astype(int) if z is not None
                   else (128, 128, 128))
        chips.append(f'<span class="chip"><span class="dot" '
                     f'style="background: rgb({r},{g},{b})"></span>{el} ({n})</span>')
    return "".join(chips)


def _cell_edges(cell, origin=None) -> list:
    """The 12 wireframe edges of a simulation cell (3x3 vectors), as point pairs.

    `origin` shifts the box so it can be drawn around a structure whose
    coordinates are not anchored at (0,0,0) — the atoms are never moved."""
    import numpy as np
    if cell is None:
        return []
    c = np.asarray(cell, dtype=float)
    if c.shape != (3, 3) or abs(np.linalg.det(c)) < 1e-6:
        return []
    o = np.zeros(3) if origin is None else np.asarray(origin, dtype=float)
    a, b, cc = c
    corners = [o + v for v in
               (0 * a, a, b, cc, a + b, a + cc, b + cc, a + b + cc)]
    pairs = [(0, 1), (0, 2), (0, 3), (1, 4), (1, 5), (2, 4),
             (2, 6), (3, 5), (3, 6), (4, 7), (5, 7), (6, 7)]
    return [[corners[i].round(4).tolist(), corners[j].round(4).tolist()]
            for i, j in pairs]


def _read_xyz(source: str | os.PathLike) -> str:
    """Accept either a path to an .xyz file or a raw .xyz string."""
    text = str(source)
    if "\n" not in text and Path(text).exists():
        return Path(text).read_text()
    return text


def _split_xyz(xyz: str, split: int, translucent: str) -> tuple[str, str]:
    """Split one .xyz text into (solid, translucent) parts at atom index `split`.

    `translucent` says which side encloses the other: "tail" (solvation shell,
    NP ligand shell — atoms after the split) or "head" (filled-tube wall —
    atoms before it)."""
    lines = xyz.strip().splitlines()
    comment, atoms = lines[1], lines[2:]
    head, tail = atoms[:split], atoms[split:]
    if translucent == "head":
        solid, trans = tail, head
    else:
        solid, trans = head, tail
    mk = lambda rows: f"{len(rows)}\n{comment}\n" + "\n".join(rows)
    return mk(solid), (mk(trans) if trans else "")


def build_html(source: str | os.PathLike, title: str = "geometry",
               width: int = 900, height: int = 640, cell=None,
               cell_origin=None, split: int | None = None,
               translucent: str = "tail", axes: list | None = None,
               theme: dict | None = None) -> str:
    """Return a self-contained HTML string that renders the given .xyz.

    `cell`/`cell_origin`: optional 3x3 cell vectors drawn as a wireframe box.
    `split`: atom count separating the two components of an ENCLOSURE system
    (solvate/fill/shell) — the enclosing part renders at 0.5 opacity;
    `translucent` = "tail" or "head" says which side that is. None = all solid.
    `axes`: gizmo arrows as [label, [x,y,z]] pairs (default cubic [100]/[010]/
    [001]); an empty label draws the arrow without text.
    """
    th = theme or THEME
    xyz = _read_xyz(source)
    if split and 0 < split < int(xyz.strip().splitlines()[0]):
        xyz_solid, xyz_trans = _split_xyz(xyz, split, translucent)
    else:
        xyz_solid, xyz_trans = xyz, ""
    first = xyz.strip().splitlines()[0].strip() if xyz.strip() else "0"
    natoms = first if first.isdigit() else "?"
    elements = {ln.split()[0] for ln in xyz.strip().splitlines()[2:] if ln.split()}
    radii = {el: sphere_radius(el) for el in sorted(elements)}
    return _TEMPLATE.substitute(
        title=title, natoms=natoms, width=width, height=height,
        bg=th["bg"], text=th["text"], cellcolor=th["cell"], outline=th["outline"],
        stick=STICK_RADIUS, opacity=ENCLOSURE_OPACITY,
        legend=_legend_html(xyz_solid, xyz_trans),
        # JSON-encode: a raw JS template literal would re-interpret escape
        # sequences in the extxyz comment (e.g. \n inside JSON-encoded
        # lt_files) and corrupt the text 3Dmol parses
        xyz_solid=json.dumps(xyz_solid),
        xyz_trans=json.dumps(xyz_trans) if xyz_trans else "null",
        edges=json.dumps(_cell_edges(cell, cell_origin)),
        radii=json.dumps(radii),
        axes=json.dumps(axes if axes is not None else DEFAULT_AXES))


def view_geometry(source: str | os.PathLike, out_html: str = "geometry.html",
                  title: str = "geometry", open_browser: bool = False) -> str:
    """Write an HTML viewer for an .xyz structure. Returns the HTML path."""
    html = build_html(source, title=title)
    out = Path(out_html)
    out.write_text(html)
    if open_browser:
        webbrowser.open(out.resolve().as_uri())
    return str(out)
