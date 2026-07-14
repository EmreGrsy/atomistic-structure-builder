"""Builder registry — the single source of truth of the backbone.

Every builder declares its parameter schema (with the question to ask when a required
value is missing) and a canonical snippet template that calls REAL functions (ASE or
mtagent). Everything else derives from here: the LLM parse prompt (catalog), the
clarifier's gap questions, the evidence packs, and the fallback snippet when no LLM
key is configured. Builder names never get hand-duplicated in prompts.

A snippet is a short Python program that must end with an ASE Atoms object bound to
the variable `atoms`. The snippet IS the artifact: it is what gets validated (Gate 1),
shown to the user, and executed (Gate 2).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

# Materials the nanoparticle builder knows how to make.
OXIDES = {"magnetite", "iron oxide", "fe3o4"}          # spinel Fe3O4 cell
# Compound SURFACES: name aliases -> slab builder in mtagent.nanostructures.
# Each cell is hand-built from its space group + experimental lattice parameters
# (ase.build.bulk knows elements only). Adding a compound = one crystal() cell +
# one line here.
TITANIA = {"titania", "tio2", "titanium dioxide", "titanium oxide", "rutile"}
SILICA = {"silica", "sio2", "silicon dioxide", "silicon oxide", "quartz"}
ZINC_OXIDE = {"zno", "zinc oxide", "zincite"}
COMPOUND_SLABS = ((TITANIA, "build_rutile_slab"),
                  (SILICA, "build_quartz_slab"),
                  (ZINC_OXIDE, "build_zincoxide_slab"),
                  (OXIDES, "build_magnetite_slab"))
# generic compounds: aliases -> COMPOUND_CRYSTALS key in nanostructures
# (checked BEFORE the legacy table so "anatase" beats the tio2 catch-all)
GENERIC_COMPOUNDS = (
    ({"anatase"}, "anatase"),
    ({"alumina", "al2o3", "aluminum oxide", "aluminium oxide", "corundum",
      "sapphire"}, "alumina"),
    ({"hematite", "fe2o3"}, "hematite"),
    ({"mgo", "magnesium oxide", "magnesia"}, "mgo"),
    ({"nio", "nickel oxide"}, "nio"),
    ({"nacl", "sodium chloride", "halite", "rock salt"}, "nacl"),
    ({"ceria", "ceo2", "cerium dioxide", "cerium oxide"}, "ceria"),
    ({"srtio3", "strontium titanate"}, "srtio3"),
    ({"pyrite", "fes2", "iron sulfide", "iron disulfide"}, "pyrite"),
    ({"gaas", "gallium arsenide"}, "gaas"),
    ({"gan", "gallium nitride"}, "gan"),
    ({"mos2", "molybdenum disulfide", "molybdenum disulphide"}, "mos2"),
)
SHEETS = (({"graphene", "graphite sheet"}, "graphene"),
          ({"h-bn", "hbn", "boron nitride", "hexagonal boron nitride"}, "hbn"))
METALS = {"gold": "Au", "silver": "Ag", "copper": "Cu", "platinum": "Pt",
          "palladium": "Pd", "nickel": "Ni", "aluminium": "Al", "aluminum": "Al"}


def slug(text: str) -> str:
    """A spec key that is also a valid Python identifier (snippets use keys as variables)."""
    s = re.sub(r"\W+", "_", str(text).strip().lower()).strip("_") or "x"
    return "_" + s if s[0].isdigit() else s


@dataclass(frozen=True)
class Param:
    name: str
    kind: str                                    # float | int | str | bool
    required: bool = False
    default: object = None
    ask: str = ""                                # question when required & missing ("" = fill silently)
    help: str = ""                               # what this parameter means (shown to the user)
    when: Callable[[dict], bool] | None = None   # only relevant if this holds for the spec


@dataclass(frozen=True)
class Builder:
    name: str
    description: str
    params: tuple[Param, ...]
    template: Callable[[dict], str]              # spec -> canonical snippet (defines `atoms`)

    def defaults(self, spec: dict) -> dict:
        """Spec with every unset parameter filled from its declared default."""
        out = dict(spec)
        for p in self.params:
            if out.get(p.name) is None and p.default is not None \
                    and (p.when is None or p.when(out)):
                out[p.name] = p.default
        return out


def _num(v, fallback) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return float(fallback)


def _is_oxide(spec: dict) -> bool:
    m = (spec.get("material") or "").lower()
    return bool(spec.get("oxide")) or any(o in m for o in OXIDES)


def _is_wulff(spec: dict) -> bool:
    return _is_oxide(spec) and (spec.get("shape") or "wulff").lower() == "wulff"


def _fcc_a(element: str) -> float | None:
    """FCC lattice constant — local table, else ASE's reference-state database."""
    from mtagent.nanoparticle import _LATTICE, _fcc_a_from_ase
    return _LATTICE.get(element) or _fcc_a_from_ase(element)


def _element_symbol(text) -> str | None:
    """Resolve free text ('gold', 'aluminum', 'Au', 'gold metal') to an element symbol."""
    from ase.data import atomic_names, atomic_numbers, chemical_symbols
    t = str(text or "").strip()
    tl = t.lower()
    names = {n.lower(): s for n, s in zip(atomic_names, chemical_symbols) if n}
    return (t if t in atomic_numbers else None) or METALS.get(tl) or names.get(tl) \
        or next((s for s in (t.capitalize(), t.upper()) if s in atomic_numbers), None) \
        or next((names[w] for w in re.findall(r"[a-z]+", tl) if w in names), None)


def normalize_material(spec: dict) -> None:
    """Reduce free-text material wording to canonical chemical identity, in place.

    "magnetite spinel" / "iron oxide (Fe3O4)"  -> material "magnetite" (oxide branch)
    "gold" / "Au" / "gold metal" / "titanium"  -> element symbol via ase.data
    Unknown oxides (e.g. "titanium dioxide") are left untouched so the template can
    raise a clear unsupported-material error instead of silently building the metal.
    """
    from ase.data import atomic_names, atomic_numbers, chemical_symbols
    m = str(spec.get("material") or spec.get("element") or "").strip()
    ml = m.lower()
    if spec.get("oxide") or any(o in ml for o in OXIDES):
        spec["material"], spec["oxide"] = "magnetite", True
        spec.pop("element", None)                 # e.g. LLM-invented "Fe3O4"
        return
    if "oxide" in ml:                             # unsupported oxide — don't guess
        return
    el = (spec.get("element") if spec.get("element") in atomic_numbers else None) \
        or _element_symbol(m)
    if el:
        spec["element"] = el
        spec["material"] = atomic_names[atomic_numbers[el]].lower()


# --------------------------- snippet templates ---------------------------------
# Each returns real, runnable code using only introspected (KG-known) functions.

def _t_nanoparticle(spec: dict) -> str:
    spec = dict(spec)
    normalize_material(spec)
    d = _num(spec.get("diameter"), 40.0)
    shape = (spec.get("shape") or ("wulff" if _is_oxide(spec) else "sphere")).lower()
    lines: list[str] = []
    if _is_oxide(spec):
        if shape in ("sphere", "cube"):
            fn = (f"build_magnetite_sphere(diameter={d:g})" if shape == "sphere"
                  else f"build_magnetite_cube(side={d:g})")
            lines += [f"from mtagent.nanostructures import {fn.split('(')[0]}",
                      f"atoms = {fn}"]
        else:      # Wulff: facet set + relative surface energies spelled out
            facets = {}
            for hkl, key, default in ((("(1, 1, 1)"), "gamma_111", 1.0),
                                      (("(1, 0, 0)"), "gamma_100", 1.4),
                                      (("(1, 1, 0)"), "gamma_110", 1.6)):
                v = spec.get(key)
                if isinstance(v, str) and v.strip().lower() in (
                        "none", "off", "no", "remove", "exclude"):
                    continue                       # facet dropped from the shape
                g = _num(v, default)
                if g <= 0:                         # "0 surface energy" = user
                    continue                       # means: no such facet
                facets[hkl] = g
            if not facets:
                raise ValueError("a Wulff particle needs at least one facet "
                                 "family — all gammas are removed (none/0)")
            gamma_src = "{" + ", ".join(f"{h}: {g:g}" for h, g in facets.items()) + "}"
            lines += ["from mtagent.wulff import build_magnetite_wulff",
                      "# relative surface energies (gamma_111 normalized to 1;",
                      "# lower gamma -> facet sits closer to the center -> larger",
                      "# facet; a facet left out of the dict is absent entirely)",
                      f"gamma = {gamma_src}",
                      f"atoms = build_magnetite_wulff(diameter={d:g}, gamma=gamma)"]
    else:
        from ase.data import atomic_numbers
        el = spec.get("element")
        if el not in atomic_numbers:
            raise ValueError(
                f"unknown nanoparticle material {spec.get('material') or el!r} — "
                f"known oxides: {sorted(OXIDES)}; metals: any element symbol "
                f"(e.g. {', '.join(sorted(set(METALS.values())))})")
        if shape == "cube":
            lines += ["from mtagent.nanostructures import build_metal_cube",
                      f'atoms = build_metal_cube("{el}", side={d:g})']
        elif _fcc_a(el) is not None:
            lines += ["from mtagent.nanoparticle import build_fcc_sphere",
                      f'atoms = build_fcc_sphere("{el}", diameter={d:g})']
        else:                          # non-FCC ground state -> generic ASE-db carve
            lines += ["from mtagent.nanostructures import build_element_sphere",
                      f'atoms = build_element_sphere("{el}", diameter={d:g})']
    n = int(spec.get("n_particles") or 1)
    if n > 1:
        gap = _num(spec.get("gap"), 10.0)
        lat = str(spec.get("lattice") or "sc").lower()
        lat_arg = f', lattice="{lat}"' if lat != "sc" else ""
        lines += ["from mtagent.cluster import build_cluster",
                  f"atoms = build_cluster(atoms, n={n}, gap={gap:g}{lat_arg})"]
    return "\n".join(lines)


def _t_nanotube(spec: dict) -> str:
    n, m = int(spec.get("n") or 6), int(spec.get("m") or 6)
    if max(n, m) < 3:                              # degenerate chirality guard
        n, m = 6, 6
    length = int(spec.get("length") or 10)
    symbol = spec.get("symbol") or "C"
    return ("from mtagent.nanostructures import build_nanotube\n"
            f'atoms = build_nanotube(n={n}, m={m}, length={length}, symbol="{symbol}")')


def _t_molecule(spec: dict) -> str:
    return ("from mtagent.pubchem import get_molecule\n"
            f'atoms = get_molecule("{spec.get("name", "water")}")')


def _t_solvent_box(spec: dict) -> str:
    mol = spec.get("molecule", "water")
    box = _num(spec.get("box_size"), 40.0)
    n = spec.get("n") or spec.get("count")         # exact molecule count wins
    if n:
        fill_arg = f", n={int(n)}"
    else:
        dens = spec.get("density")
        fill_arg = f", density={_num(dens, 1.0):g}" if dens is not None else ""
    return ("from mtagent.pubchem import get_molecule\n"
            "from mtagent.solvent import build_solvent_box\n"
            f'mol = get_molecule("{mol}")\n'
            f'atoms = build_solvent_box(mol, box_size={box:g}{fill_arg}, name="{mol}")')


# compound bulk CELLS (same alias sets as the slabs; fn module noted per entry)
COMPOUND_CELLS = ((TITANIA, "nanostructures", "build_rutile"),
                  (SILICA, "nanostructures", "build_quartz"),
                  (ZINC_OXIDE, "nanostructures", "build_zincoxide"),
                  (OXIDES, "wulff", "build_spinel"))


def _t_bulk(spec: dict) -> str:
    material = str(spec.get("element") or spec.get("material") or "")
    rep = re.findall(r"\d+", str(spec.get("repeat") or "2x2x2"))
    rep = (rep + rep[-1:] * 2)[:3] if rep else ["2", "2", "2"]
    rep_t = "(" + ", ".join(rep) + ")"
    ml = material.lower()
    for names, kind in SHEETS:              # "a graphene sheet" parses as bulk
        if any(t in ml for t in names):
            return ("from mtagent.nanostructures import build_sheet\n"
                    f'atoms = build_sheet("{kind}", width=25, vacuum=10)')
    for names, key in GENERIC_COMPOUNDS:
        if any(t in ml for t in names):
            return ("from mtagent.nanostructures import build_compound\n"
                    f'atoms = build_compound("{key}").repeat({rep_t})')
    for names, mod, fn in COMPOUND_CELLS:
        if any(t in ml for t in names):
            return (f"from mtagent.{mod} import {fn}\n"
                    f"atoms = {fn}().repeat({rep_t})")
    el = _element_symbol(material)
    if el is None:
        supported = ", ".join(sorted(
            [min(names, key=len) for names, _, _ in COMPOUND_CELLS]
            + [min(names, key=len) for names, _ in GENERIC_COMPOUNDS]))
        raise ValueError(f"unknown bulk material {material!r} — supported: any "
                         f"element (name or symbol), or {supported}")
    return ("from mtagent.nanostructures import build_element_bulk\n"
            f'atoms = build_element_bulk("{el}", repeat={rep_t})')


def _t_surface_slab(spec: dict) -> str:
    material = str(spec.get("element") or spec.get("material") or "")
    digits = [int(c) for c in re.findall(r"\d", str(spec.get("miller") or "111"))]
    if len(digits) == 4:            # hexagonal Miller-Bravais (hkil): i = -(h+k)
        digits = [digits[0], digits[1], digits[3]]
    miller = tuple(digits[:3]) if any(digits[:3]) else (1, 1, 1)
    rep = re.search(r"(\d+)\s*[x×]\s*(\d+)", str(spec.get("repeat") or ""))
    plane = (f"nx={int(rep.group(1))}, ny={int(rep.group(2))}" if rep
             else f"width={_num(spec.get('width'), 25.0):g}")
    dims = (f"thickness={_num(spec.get('thickness'), 10.0):g}, {plane}, "
            f"vacuum={_num(spec.get('vacuum'), 10.0):g}")
    ml = material.lower()
    for names, kind in SHEETS:                      # 2D sheets have no Miller
        if any(t in ml for t in names):
            return ("from mtagent.nanostructures import build_sheet\n"
                    f'atoms = build_sheet("{kind}", '
                    f"width={_num(spec.get('width'), 25.0):g}, "
                    f"vacuum={_num(spec.get('vacuum'), 10.0):g})")
    for names, key in GENERIC_COMPOUNDS:
        if any(t in ml for t in names):
            return ("from mtagent.nanostructures import build_compound_slab\n"
                    f'atoms = build_compound_slab("{key}", miller={miller}, {dims})')
    for names, fn in COMPOUND_SLABS:
        if any(t in ml for t in names):
            return (f"from mtagent.nanostructures import {fn}\n"
                    f"atoms = {fn}(miller={miller}, {dims})")
    el = _element_symbol(material)
    if el is None:
        supported = sorted(
            [min(names, key=len) for names, _ in COMPOUND_SLABS]
            + [min(names, key=len) for names, _ in GENERIC_COMPOUNDS]
            + [min(names, key=len) for names, _ in SHEETS])
        raise ValueError(f"unknown slab material {material!r} — supported: any "
                         f"element (name or symbol), or {', '.join(supported)}")
    return ("from mtagent.nanostructures import build_slab\n"
            f'atoms = build_slab("{el}", miller={miller}, {dims})')


# ------------------------------- the registry ----------------------------------

BUILDERS: dict[str, Builder] = {b.name: b for b in (
    Builder(
        "nanoparticle", "Metal (FCC) or oxide (magnetite spinel) nanoparticle; "
        "optional -OH surface capping and multi-NP cluster (deterministic Moltemplate lattice).",
        (Param("material", "str", required=True,
               ask="Which material is the nanoparticle? (e.g. magnetite, gold)",
               help="chemical identity — magnetite (Fe3O4 spinel) or an FCC metal (gold, silver, copper, platinum…)"),
         Param("diameter", "float", required=True, default=40.0,
               ask="What diameter, in Å? (1 nm = 10 Å; default 40 Å)",
               help="particle size in Å (1 nm = 10 Å); the carve targets this extent"),
         Param("shape", "str", required=True,
               ask="Shape — wulff (faceted), sphere, or cube? (if the user says "
                   "'default' or doesn't care: answer 'wulff' for oxides, "
                   "'sphere' for metals)",
               help="wulff = equilibrium facets from literature surface energies (oxides); sphere = spherical carve; cube = {100} cube"),
         Param("gamma_111", "float", default=1.0, when=_is_wulff,
               help="relative surface energy of the {111} facet (normalized to 1; "
                    "lower = larger facet in the Wulff shape; 'none' removes the "
                    "facet from the shape entirely)"),
         Param("gamma_100", "float", default=1.4, when=_is_wulff,
               help="relative surface energy of the {100} facet (literature ordering "
                    "{111} < {100} < {110} for magnetite; 'none' removes the facet)"),
         Param("gamma_110", "float", default=1.6, when=_is_wulff,
               help="relative surface energy of the {110} facet ('none' or 0 removes "
                    "the facet — e.g. for a {111}+{100}-only cuboctahedral particle)"),
         Param("n_particles", "int", required=True, default=1,
               ask="A single nanoparticle, or a cluster — how many? (default 1)",
               help="1 = single particle; >1 = deterministic cluster assembled via Moltemplate (compact lattice, spacing = diameter + gap)"),
         Param("gap", "float", default=10.0,
               help="surface-to-surface spacing between cluster particles, Å"),
         Param("lattice", "str", default="sc",
               help="superlattice packing of a multi-particle supercrystal: "
                    "sc (simple cubic), fcc, bcc, or an explicit close-packed "
                    "stacking sequence like ABCABCABAB (fcc with a stacking "
                    "fault); complete-cell counts get a periodic cell"),
         Param("element", "str",
               help="chemical symbol; derived from the material for metals (e.g. Au)")),
        _t_nanoparticle),
    Builder(
        "nanotube", "Carbon (or other) nanotube of chirality (n, m).",
        (Param("n", "int", required=True, default=6,
               ask="Nanotube chirality (n, m)? (default (6,6), metallic armchair)",
               help="chirality index n — (n,n) armchair is metallic, (n,0) zigzag; radius grows with n (a (6,6) CNT is ~8 Å wide)"),
         Param("m", "int", default=6, help="chirality index m (see n)"),
         Param("length", "int", required=True, default=10,
               ask="Nanotube length, in unit cells? (default 10 ≈ 25 Å)",
               help="length in unit cells; 1 cell ≈ 2.46 Å for carbon"),
         Param("symbol", "str", default="C", help="wall element (C = carbon nanotube)")),
        _t_nanotube),
    Builder(
        "molecule", "A single molecule with real 3D coordinates from PubChem.",
        (Param("name", "str", required=True, ask="Which molecule?",
               help="compound name, resolved on PubChem (real 3D coordinates, never invented)"),
         Param("count", "int",
               help="copies to place during assembly (default: auto — fill at liquid density)")),
        _t_molecule),
    Builder(
        "solvent_box", "Periodic box filled with a liquid at its reference density (packmol).",
        (Param("molecule", "str", required=True, ask="Which solvent liquid?",
               help="the liquid filling the box (molecule fetched from PubChem)"),
         Param("box_size", "float", required=True,
               ask="Solvent box edge, in Å? (default: auto — sized to encompass the solute + 20 Å)",
               help="cubic box edge in Å; auto-sized so the box encompasses the solute + 20 Å"),
         Param("n", "int",
               help="exact number of solvent molecules — overrides the density fill"),
         Param("density", "float",
               help="g/cm³; defaults to the liquid's literature reference density")),
        _t_solvent_box),
    Builder(
        "bulk", "Bulk crystal (periodic supercell of the conventional cell — "
        "no surfaces, no vacuum): any element or magnetite/titania/silica/zinc oxide.",
        (Param("material", "str", required=True,
               ask="Which material for the bulk crystal?",
               help="any element (name or symbol) or magnetite / titania (rutile) "
                    "/ silica (alpha-quartz) / zinc oxide (wurtzite)"),
         Param("repeat", "str", default="2x2x2",
               help="supercell as NxMxK repeats of the conventional cell "
                    "(e.g. 3x3x3; a single number N means NxNxN)")),
        _t_bulk),
    Builder(
        "surface_slab", "Metal surface slab from bulk (Miller termination, "
        "thickness and width in Angstrom, vacuum).",
        (Param("element", "str", required=True, ask="Which material for the surface?",
               help="the material, as name or symbol (any element, e.g. aluminum, "
                    "Au — or magnetite / titania / silica / zinc oxide)"),
         Param("miller", "str", required=True, default="111",
               ask="Which surface termination (Miller indices — e.g. 111, 001, 110)? "
                   "(default 111)",
               help="crystallographic termination, e.g. 111 (close-packed), 001, 110"),
         Param("thickness", "float", default=10.0,
               help="minimum slab thickness, Å (layers are added until reached)"),
         Param("repeat", "str",
               help="in-plane supercell as NxM (e.g. 3x4) — the surface-science "
                    "convention; overrides width when given"),
         Param("width", "float", default=25.0,
               help="minimum in-plane extent, Å (cell repeated laterally until "
                    "reached; used only when no NxM repeat is given)"),
         Param("vacuum", "float", default=10.0, help="vacuum padding above the surface, Å")),
        _t_surface_slab),
)}


def catalog() -> str:
    """Compact builder catalog for LLM prompts — derived, never hand-written."""
    lines = []
    for b in BUILDERS.values():
        ps = ", ".join(f"{p.name}:{p.kind}" + (f"={p.default}" if p.default is not None else "")
                       for p in b.params)
        lines.append(f"- {b.name}: {b.description}  spec fields: {{{ps}}}")
    return "\n".join(lines)
