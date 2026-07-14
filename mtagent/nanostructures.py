"""Additional builders: nanotubes, generic spherical NPs, surface slabs, and interfaces.

All ASE-based. Spherical carve works for ANY crystal cell (metals via nanoparticle.py,
oxides via the spinel cell here). Interfaces stack two slabs (lattice-matching is the
caller's responsibility; a mismatch raises).
"""
from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import nanotube, bulk, surface, stack

from .wulff import build_spinel


# --- nanotube ----------------------------------------------------------------

def build_nanotube(n: int = 6, m: int = 6, length: int = 10,
                   symbol: str = "C", bond: float = 1.42) -> Atoms:
    """Carbon (or other) nanotube of chirality (n, m), `length` unit cells."""
    cnt = nanotube(n, m, length=length, bond=bond, symbol=symbol)
    cnt.set_pbc([False, False, True])
    cnt.info["provenance"] = {"source": "ASE", "type": "nanotube", "chirality": [n, m],
                              "length": length, "symbol": symbol, "n_atoms": len(cnt)}
    return cnt


# --- generic spherical carve -------------------------------------------------

def _no_empty(atoms: Atoms, what: str, size: float) -> Atoms:
    if len(atoms) == 0:
        raise ValueError(f"{what} of {size:g} Å came out EMPTY — the size is "
                         "smaller than one unit cell. Sizes are in Ångström "
                         "(1 nm = 10 Å); use a larger value.")
    return atoms


def carve_sphere(cell: Atoms, diameter: float) -> Atoms:
    """Carve a sphere of the given diameter (A) from any (periodic) crystal cell."""
    a = float(cell.cell.lengths().max())
    nrep = int(np.ceil(diameter / a)) + 2
    sc = cell.repeat((nrep, nrep, nrep))
    sc.set_positions(sc.get_positions() - sc.get_positions().mean(axis=0))
    sphere = sc[np.linalg.norm(sc.get_positions(), axis=1) <= diameter / 2.0]
    sphere.set_cell(None)
    sphere.set_pbc(False)
    symbols = sphere.get_chemical_symbols()
    sphere.info["provenance"] = {"source": "ASE", "type": "spherical_nanoparticle",
                                 "diameter": diameter, "n_atoms": len(sphere),
                                 "composition": {e: symbols.count(e) for e in set(symbols)}}
    return _no_empty(sphere, "spherical carve", diameter)


def build_magnetite_sphere(diameter: float = 40.0) -> Atoms:
    """A plain spherical (non-faceted) magnetite NP — alternative to the Wulff shape."""
    return carve_sphere(build_spinel(), diameter)


def build_element_sphere(element: str, diameter: float = 40.0) -> Atoms:
    """Spherical NP of ANY element in its ground-state crystal structure.

    Lattice type and constants come from ASE's built-in reference-state database
    (ase.build.bulk with no arguments beyond the element) — no hand-kept tables.
    """
    try:
        cell = bulk(element, cubic=True)
    except (ValueError, RuntimeError):   # hcp etc. have no cubic setting
        cell = bulk(element, orthorhombic=True)
    return carve_sphere(cell, diameter)


# --- generic cube carve ------------------------------------------------------

def carve_cube(cell: Atoms, side: float) -> Atoms:
    """Carve a cube of the given side length (A) from any (periodic) crystal cell."""
    a = float(cell.cell.lengths().max())
    nrep = int(np.ceil(side / a)) + 2
    sc = cell.repeat((nrep, nrep, nrep))
    sc.set_positions(sc.get_positions() - sc.get_positions().mean(axis=0))
    mask = (np.abs(sc.get_positions()) <= side / 2.0).all(axis=1)
    cube = sc[mask]
    cube.set_cell(None)
    cube.set_pbc(False)
    symbols = cube.get_chemical_symbols()
    cube.info["provenance"] = {"source": "ASE", "type": "cubic_nanoparticle",
                               "side": side, "n_atoms": len(cube),
                               "composition": {e: symbols.count(e) for e in set(symbols)}}
    return cube


def build_metal_cube(element: str = "Au", side: float = 40.0,
                     crystal: str = "fcc", a: float | None = None) -> Atoms:
    """A cubic metal nanoparticle (carved from bulk, {100} faces)."""
    if a is None:                          # ASE demands `a` with an explicit
        from .nanoparticle import _LATTICE, _fcc_a_from_ase  # crystalstructure
        a = _LATTICE.get(element) or _fcc_a_from_ase(element)
    b = bulk(element, crystal, a=a, cubic=True) if a else bulk(element, crystal, cubic=True)
    return carve_cube(b, side)


def build_magnetite_cube(side: float = 40.0) -> Atoms:
    """A cubic magnetite NP ({100} faces) — carved, not Wulff."""
    return carve_cube(build_spinel(), side)


# --- surface slab ------------------------------------------------------------

def _ext_gcd(a: int, b: int) -> tuple[int, int]:
    if b == 0:
        return 1, 0
    if a % b == 0:
        return 0, 1
    x, y = _ext_gcd(b, a % b)
    return y, x - y * (a // b)


def _inplane_uvw(cell, miller) -> list:
    """Crystallographic direction indices [uvw] of the two in-plane surface-cell
    vectors, in units of the bulk cell axes — the same integer basis
    ase.build.surface() constructs internally (it just doesn't report it)."""
    from math import gcd
    h, k, l = (int(m) for m in miller)
    nonzero = [i for i, m in enumerate((h, k, l)) if m != 0]
    if len(nonzero) == 1:
        return {0: [[0, 1, 0], [0, 0, 1]],
                1: [[0, 0, 1], [1, 0, 0]],
                2: [[1, 0, 0], [0, 1, 0]]}[nonzero[0]]
    a1, a2, a3 = np.asarray(cell, dtype=float)
    p, q = _ext_gcd(k, l)
    k1 = np.dot(p * (k * a1 - h * a2) + q * (l * a1 - h * a3), l * a2 - k * a3)
    k2 = np.dot(l * (k * a1 - h * a2) - k * (l * a1 - h * a3), l * a2 - k * a3)
    if abs(k2) > 1e-10:
        i = -int(round(k1 / k2))
        p, q = p + i * l, q - i * k
    c1 = [p * k + q * l, -p * h, -q * h]
    g1 = gcd(gcd(abs(c1[0]), abs(c1[1])), abs(c1[2])) or 1
    c1 = [v // g1 for v in c1]
    g2 = abs(gcd(l, k)) or 1
    c2 = [0, l // g2, -k // g2]
    return [c1, c2]


def _slab_from_cell(cell: Atoms, miller, thickness: float, width: float,
                    vacuum: float, layers: int | None = None,
                    nx: int | None = None, ny: int | None = None) -> Atoms:
    """Cut a slab from ANY bulk cell: >= `thickness` (Å) thick (layers grown until
    reached; `layers` overrides). In-plane size: `nx` x `ny` supercell if given
    (the surface-science convention), else repeated until >= `width` (Å)."""
    if layers is None:
        layers = 1
        while layers < 120:
            s = surface(cell, miller, layers, vacuum=None)
            z = s.get_positions()[:, 2]
            if z.max() - z.min() >= thickness:
                break
            layers += 1
    slab = surface(cell, miller, layers, vacuum=vacuum)
    lx, ly = slab.cell.lengths()[:2]
    if nx is None:
        nx = max(1, int(np.ceil(width / lx)))
    if ny is None:
        ny = max(1, int(np.ceil(width / ly)))
    slab = slab.repeat((int(nx), int(ny), 1))
    z = slab.get_positions()[:, 2]
    symbols = slab.get_chemical_symbols()
    slab.info["provenance"] = {
        "source": "ASE", "type": "surface_slab",
        "miller": list(miller), "layers": layers,
        "in_plane_uvw": _inplane_uvw(cell.cell[:], miller),
        "repeat": [int(nx), int(ny)],
        "thickness_A": round(float(z.max() - z.min()), 2),
        "in_plane_A": [round(float(v), 2) for v in slab.cell.lengths()[:2]],
        "vacuum": vacuum, "n_atoms": len(slab),
        "composition": {e: symbols.count(e) for e in set(symbols)}}
    return slab


# diamond-structure elements ASE's reference db won't build unaided
# (experimental lattice constants, Å)
_DIAMOND_A = {"C": 3.567, "Si": 5.431, "Ge": 5.658}


def _element_bulk_cell(element: str, crystal: str | None = None,
                       a: float | None = None) -> Atoms:
    """Elemental bulk cell, CONVENTIONAL where one exists (Miller indices and
    crystallographic directions are read in the given cell's basis — ASE's
    primitive fcc/bcc cell would silently redefine them). Accepts element
    names ('platinum') as well as symbols."""
    from ase.data import atomic_numbers
    if element not in atomic_numbers:      # LLM snippets pass names ('platinum')
        from .registry import _element_symbol
        element = _element_symbol(element) or element
    try:
        return bulk(element, crystal, a=a, cubic=True) if crystal \
            else bulk(element, cubic=True)
    except (ValueError, RuntimeError):
        try:                                    # non-cubic (e.g. hcp)
            return bulk(element, crystal, a=a) if crystal else bulk(element)
        except (ValueError, RuntimeError):
            if element in _DIAMOND_A:           # ASE's db can't infer these
                return bulk(element, "diamond", a=a or _DIAMOND_A[element],
                            cubic=True)
            raise


def build_element_bulk(element: str = "Au", repeat=(2, 2, 2),
                       crystal: str | None = None, a: float | None = None) -> Atoms:
    """Bulk crystal of any element: conventional cell repeated to an NxMxK
    supercell — fully periodic, no vacuum, no surfaces."""
    cell = _element_bulk_cell(element, crystal, a)
    rep = tuple(int(v) for v in repeat)
    atoms = cell.repeat(rep)
    symbols = atoms.get_chemical_symbols()
    atoms.info["provenance"] = {
        "source": "ASE", "type": "bulk_crystal", "element": element,
        "repeat": list(rep),
        "cell_A": [round(float(v), 3) for v in atoms.cell.lengths()],
        "n_atoms": len(atoms),
        "composition": {e: symbols.count(e) for e in set(symbols)}}
    return atoms


def build_slab(element: str = "Au", miller=(1, 1, 1), thickness: float = 10.0,
               width: float = 25.0, vacuum: float = 10.0, layers: int | None = None,
               nx: int | None = None, ny: int | None = None,
               crystal: str | None = None, a: float | None = None) -> Atoms:
    """Elemental surface slab: at least `thickness` (Å) thick and `width` (Å) wide,
    with the chosen Miller termination (e.g. (1,1,1), (0,0,1), (1,1,0)).

    Lattice type/constants come from ASE's reference-state database unless
    `crystal`/`a` override them. `layers` (if given) overrides `thickness`.

    The bulk MUST be the CONVENTIONAL cubic cell where one exists: Miller
    indices are interpreted in the given cell's basis, and with ASE's default
    primitive fcc/bcc cell "(001)" would silently cut a {111} plane.
    """
    b = _element_bulk_cell(element, crystal, a)
    slab = _slab_from_cell(b, miller, thickness, width, vacuum, layers, nx, ny)
    slab.info["provenance"]["element"] = element
    return slab


def build_magnetite_slab(miller=(0, 0, 1), thickness: float = 10.0,
                         width: float = 25.0, vacuum: float = 10.0,
                         layers: int | None = None,
                         nx: int | None = None, ny: int | None = None) -> Atoms:
    """Magnetite (Fe3O4 spinel) surface slab with the chosen Miller termination.

    Geometry only: the cut is a plain crystal truncation — polar-surface
    reconstructions/terminations are not modeled.
    """
    slab = _slab_from_cell(build_spinel(), miller, thickness, width, vacuum, layers, nx, ny)
    slab.info["provenance"]["material"] = "magnetite"
    return slab


def build_rutile(a: float = 4.593, c: float = 2.959, u: float = 0.305) -> Atoms:
    """Rutile TiO2 conventional cell (P4_2/mnm #136; experimental a=4.593 A,
    c=2.959 A, u=0.305 — Abrahams & Bernstein 1971)."""
    from ase.spacegroup import crystal
    return crystal(["Ti", "O"], basis=[(0, 0, 0), (u, u, 0)], spacegroup=136,
                   cellpar=[a, a, c, 90, 90, 90])


def build_rutile_slab(miller=(1, 1, 0), thickness: float = 10.0,
                      width: float = 25.0, vacuum: float = 10.0,
                      layers: int | None = None,
                      nx: int | None = None, ny: int | None = None) -> Atoms:
    """Rutile TiO2 surface slab ((110) is the standard stable face).

    Geometry only: plain crystal truncation — no surface reconstruction.
    """
    slab = _slab_from_cell(build_rutile(), miller, thickness, width, vacuum, layers, nx, ny)
    slab.info["provenance"]["material"] = "rutile TiO2"
    return slab


def build_quartz(a: float = 4.9134, c: float = 5.4052) -> Atoms:
    """Alpha-quartz SiO2 conventional cell (P3_121, experimental parameters
    from Levien et al. 1980)."""
    from ase.spacegroup import crystal
    return crystal(["Si", "O"],
                   basis=[(0.4697, 0.0, 1 / 3), (0.4135, 0.2669, 0.1191)],
                   spacegroup=152, cellpar=[a, a, c, 90, 90, 120])


def build_quartz_slab(miller=(0, 0, 1), thickness: float = 10.0,
                      width: float = 25.0, vacuum: float = 10.0,
                      layers: int | None = None,
                      nx: int | None = None, ny: int | None = None) -> Atoms:
    """Alpha-quartz SiO2 surface slab (geometry only, no reconstruction)."""
    slab = _slab_from_cell(build_quartz(), miller, thickness, width, vacuum, layers, nx, ny)
    slab.info["provenance"]["material"] = "alpha-quartz SiO2"
    return slab


def build_zincoxide(a: float = 3.2495, c: float = 5.2069, u: float = 0.3817) -> Atoms:
    """Wurtzite ZnO conventional cell (P6_3mc #186; experimental a=3.2495 A,
    c=5.2069 A, u=0.3817)."""
    from ase.spacegroup import crystal
    return crystal(["Zn", "O"],
                   basis=[(1 / 3, 2 / 3, 0.0), (1 / 3, 2 / 3, u)],
                   spacegroup=186, cellpar=[a, a, c, 90, 90, 120])


def build_zincoxide_slab(miller=(1, 0, 0), thickness: float = 10.0,
                         width: float = 25.0, vacuum: float = 10.0,
                         layers: int | None = None,
                         nx: int | None = None, ny: int | None = None) -> Atoms:
    """Wurtzite ZnO surface slab ((100) = the nonpolar m-plane; (001) is polar).

    Geometry only: plain crystal truncation — no reconstruction/charge passivation.
    """
    slab = _slab_from_cell(build_zincoxide(), miller, thickness, width, vacuum, layers, nx, ny)
    slab.info["provenance"]["material"] = "wurtzite ZnO"
    return slab


# --- generic compound crystals (one literature space-group cell each) --------
# experimental lattice parameters from the classic structure determinations
COMPOUND_CRYSTALS: dict = {
    "anatase":   dict(symbols=["Ti", "O"], sg=141,      # setting 1: 12-atom
                      basis=[(0, 0, 0), (0, 0, 0.2081)],  # cell, Ti-O 1.93 A
                      cellpar=[3.7842, 3.7842, 9.5146, 90, 90, 90],
                      label="anatase TiO2"),
    "alumina":   dict(symbols=["Al", "O"], sg=167,
                      basis=[(0, 0, 0.35216), (0.30624, 0, 0.25)],
                      cellpar=[4.7602, 4.7602, 12.9933, 90, 90, 120],
                      label="alpha-Al2O3 (corundum)"),
    "hematite":  dict(symbols=["Fe", "O"], sg=167,
                      basis=[(0, 0, 0.3553), (0.3059, 0, 0.25)],
                      cellpar=[5.038, 5.038, 13.772, 90, 90, 120],
                      label="alpha-Fe2O3 (hematite)"),
    "mgo":       dict(symbols=["Mg", "O"], sg=225,
                      basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                      cellpar=[4.212, 4.212, 4.212, 90, 90, 90],
                      label="MgO (rock salt)"),
    "nio":       dict(symbols=["Ni", "O"], sg=225,
                      basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                      cellpar=[4.177, 4.177, 4.177, 90, 90, 90],
                      label="NiO (rock salt)"),
    "nacl":      dict(symbols=["Na", "Cl"], sg=225,
                      basis=[(0, 0, 0), (0.5, 0.5, 0.5)],
                      cellpar=[5.640, 5.640, 5.640, 90, 90, 90],
                      label="NaCl (halite)"),
    "ceria":     dict(symbols=["Ce", "O"], sg=225,
                      basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                      cellpar=[5.411, 5.411, 5.411, 90, 90, 90],
                      label="CeO2 (fluorite)"),
    "srtio3":    dict(symbols=["Sr", "Ti", "O"], sg=221,
                      basis=[(0, 0, 0), (0.5, 0.5, 0.5), (0, 0.5, 0.5)],
                      cellpar=[3.905, 3.905, 3.905, 90, 90, 90],
                      label="SrTiO3 (perovskite)"),
    "pyrite":    dict(symbols=["Fe", "S"], sg=205,
                      basis=[(0, 0, 0), (0.3851, 0.3851, 0.3851)],
                      cellpar=[5.417, 5.417, 5.417, 90, 90, 90],
                      label="FeS2 (pyrite)"),
    "gaas":      dict(symbols=["Ga", "As"], sg=216,
                      basis=[(0, 0, 0), (0.25, 0.25, 0.25)],
                      cellpar=[5.6535, 5.6535, 5.6535, 90, 90, 90],
                      label="GaAs (zincblende)"),
    "gan":       dict(symbols=["Ga", "N"], sg=186,
                      basis=[(1 / 3, 2 / 3, 0), (1 / 3, 2 / 3, 0.377)],
                      cellpar=[3.189, 3.189, 5.185, 90, 90, 120],
                      label="GaN (wurtzite)"),
    "mos2":      dict(symbols=["Mo", "S"], sg=194,
                      basis=[(1 / 3, 2 / 3, 0.25), (1 / 3, 2 / 3, 0.627)],
                      cellpar=[3.160, 3.160, 12.294, 90, 90, 120],
                      label="2H-MoS2"),
}


def _compound_key(name: str) -> str:
    """Resolve any alias/formula ('GaAs', 'MgO', 'magnesia') to the table key."""
    n = str(name).strip().lower()
    if n in COMPOUND_CRYSTALS:
        return n
    from .registry import GENERIC_COMPOUNDS
    for aliases, key in GENERIC_COMPOUNDS:
        if n in aliases or any(a in n for a in aliases):
            return key
    raise KeyError(f"unknown compound {name!r} — known: "
                   f"{sorted(COMPOUND_CRYSTALS)}")


def build_compound(name: str) -> Atoms:
    """Conventional cell of a named compound from COMPOUND_CRYSTALS."""
    from ase.spacegroup import crystal
    d = COMPOUND_CRYSTALS[_compound_key(name)]
    kw = {"setting": d["setting"]} if "setting" in d else {}
    return crystal(d["symbols"], basis=d["basis"], spacegroup=d["sg"],
                   cellpar=d["cellpar"], **kw)


def build_compound_slab(name: str, miller=(0, 0, 1), thickness: float = 10.0,
                        width: float = 25.0, vacuum: float = 10.0,
                        layers: int | None = None,
                        nx: int | None = None, ny: int | None = None) -> Atoms:
    """Surface slab of a named compound (plain truncation, no reconstruction)."""
    key = _compound_key(name)
    slab = _slab_from_cell(build_compound(key), miller, thickness, width,
                           vacuum, layers, nx, ny)
    slab.info["provenance"]["material"] = COMPOUND_CRYSTALS[key]["label"]
    return slab


def build_sheet(kind: str = "graphene", width: float = 25.0,
                vacuum: float = 10.0) -> Atoms:
    """A 2D sheet: graphene or hexagonal boron nitride (basal plane)."""
    from ase.build import graphene
    if kind == "graphene":
        sheet = graphene(vacuum=vacuum)
    elif kind in ("hbn", "h-bn", "boron nitride"):
        sheet = graphene(formula="BN", a=2.504, vacuum=vacuum)
    else:
        raise ValueError(f"unknown sheet {kind!r} — graphene or hbn")
    lx, ly = sheet.cell.lengths()[:2]
    sheet = sheet.repeat((max(1, int(np.ceil(width / lx))),
                          max(1, int(np.ceil(width / ly))), 1))
    symbols = sheet.get_chemical_symbols()
    sheet.info["provenance"] = {
        "source": "ASE", "type": "sheet", "material": kind,
        "n_atoms": len(sheet), "vacuum": vacuum,
        "composition": {e: symbols.count(e) for e in set(symbols)}}
    return sheet


# --- interface (stack two slabs) --------------------------------------------

def build_interface(slab1: Atoms, slab2: Atoms, axis: int = 2,
                    distance: float = 2.5, maxstrain: float = 0.5) -> Atoms:
    """Stack two slabs into an interface along `axis`. Requires compatible in-plane cells."""
    iface = stack(slab1, slab2, axis=axis, distance=distance, maxstrain=maxstrain)
    iface.info["provenance"] = {"source": "ASE", "type": "interface",
                                "n_atoms": len(iface), "axis": axis, "distance": distance}
    return iface
