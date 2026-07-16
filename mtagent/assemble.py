"""Relation-driven assembly — combine built constituents into one showcase cell.

The relation between constituents is part of the parsed spec (water INSIDE a nanotube
is a different structure than water AROUND one):

  inside     guest molecules packed inside the host cavity (packmol, host fixed)
  around     host solvated: carve a cavity in a solvent box, insert the host
  coated_by  guest molecules packed as a shell, then assembled via Moltemplate
             (molecule defined once as .lt, instantiated N times; Gate 1 + Gate 2)
  on         guest placed on the host surface (adsorption geometry)

The combined cell is a SHOWCASE: geometry only, not equilibrated. A real simulation
cell needs MD equilibration, which is out of scope for the Moltemplate agent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io import read, write

from .kg import MoltemplateKG
from .moltemplate_emit import emit_assembly
from .packing import find_packmol, pack_shell
from .runner import run_moltemplate, parse_lammps_data
from .solvent import SOLVENT_DENSITY, _NA

RELATIONS = {
    "inside": "guest molecules packed INSIDE the host (e.g. methanol inside a nanotube)",
    "around": "host SOLVATED by the guest liquid (solvent box carved, host inserted)",
    "coated_by": "guest molecules packed as a SHELL coating the host surface",
    "on": "guest placed ON the host surface (adsorption geometry)",
    "between": "guest liquid film SANDWICHED between two copies of the host slab "
               "(confined film, e.g. water between two magnetite slabs)",
}


def relation_catalog() -> str:
    return "\n".join(f"- {k}: {v}" for k, v in RELATIONS.items())


def extent(atoms: Atoms) -> float:
    """Largest side of the axis-aligned bounding box (for sizing solvent boxes)."""
    p = atoms.get_positions()
    return float((p.max(axis=0) - p.min(axis=0)).max())


def _centered(atoms: Atoms) -> Atoms:
    a = atoms.copy()
    a.set_positions(a.get_positions() - a.get_positions().mean(axis=0))
    return a


def fill_inside(host: Atoms, guest: Atoms, n: int | None = None, clearance: float = 1.7,
                density: float | None = None, tolerance: float = 2.0,
                workdir: str | Path = "data/work/inside", timeout: int = 900) -> Atoms:
    """Pack `n` copies of `guest` inside `host` (packmol; host fixed at the origin).

    Tubular hosts (extent along z >> lateral) use an inside-cylinder constraint of
    radius (tube radius - clearance); other hosts an inside-box of the shrunk bounding
    box. n=None fills at ~60% of the guest liquid's reference density (a physically
    sensible showcase load, not an equilibrated one).
    """
    import subprocess
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")

    if is_framework(host):
        # a MOF's pores are the whole cell, not a bounding box: shrinking the
        # box would seal the borders and lose most of the pore network
        return fill_pores(host, guest, n=n, density=density,
                          tolerance=tolerance, workdir=workdir, timeout=timeout)

    host = _centered(host)
    p = host.get_positions()
    lo, hi = p.min(axis=0), p.max(axis=0)
    r_lat = float(np.sqrt(p[:, 0] ** 2 + p[:, 1] ** 2).max())
    tubular = (host.info.get("provenance", {}).get("type") == "nanotube"
               or (hi[2] - lo[2]) > 1.5 * (2 * r_lat))

    if tubular:
        r_in = r_lat - clearance
        if r_in < 1.0:
            raise ValueError(f"host cavity too narrow to fill (free radius {r_in:.2f} Å)")
        length = (hi[2] - lo[2]) - 1.0
        volume_A3 = np.pi * r_in ** 2 * length
        constraint = (f"  inside cylinder 0. 0. {lo[2] + 0.5:.3f} 0. 0. 1. "
                      f"{r_in:.3f} {length:.3f}\n")
    else:
        inner_lo, inner_hi = lo + clearance, hi - clearance
        if (inner_hi - inner_lo).min() < 2.0:
            raise ValueError("host cavity too small to fill")
        volume_A3 = float(np.prod(inner_hi - inner_lo))
        constraint = ("  inside box "
                      + " ".join(f"{v:.3f}" for v in (*inner_lo, *inner_hi)) + "\n")

    if n is None:
        name = str(guest.info.get("provenance", {}).get("query", "")).lower()
        rho = density or SOLVENT_DENSITY.get(name, 0.8)
        molar = float(guest.get_masses().sum())
        n = max(1, int(0.6 * rho * (volume_A3 * 1e-24) * _NA / molar))
        # one packmol run converges up to ~8000 guest atoms; an auto fill
        # beyond that is capped (a partial showcase fill, recorded in info)
        n = min(n, max(1, 8000 // max(1, len(guest))))
    elif int(n) * len(guest) > 10000:
        raise RuntimeError(
            f"that would pack {int(n) * len(guest)} guest atoms — beyond the "
            "current packing limit (~10,000; packmol converges too slowly for "
            "one huge fill). Use a smaller count, or build a smaller system "
            "and replicate it.")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    host_xyz, guest_xyz, out_xyz = work / "host.xyz", work / "guest.xyz", work / "filled.xyz"
    write(str(host_xyz), host, format="xyz")
    write(str(guest_xyz), _centered(guest), format="xyz")
    inp = work / "fill.inp"
    inp.write_text(
        f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
        f"structure {host_xyz.name}\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"structure {guest_xyz.name}\n  number {n}\n{constraint}end structure\n")
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol fill_inside failed:\n{proc.stdout[-1200:]}")

    filled = read(str(out_xyz))
    filled.info["packmol_inp"] = inp.read_text()
    filled.info["assembly"] = {"relation": "inside", "n_guests": n,
                               "host_atoms": len(host), "guest_atoms": len(guest),
                               "cavity": "cylinder" if tubular else "box",
                               "free_volume_A3": round(volume_A3, 1)}
    return filled


def is_framework(atoms: Atoms) -> bool:
    """A porous periodic host (MOF): a filled 3D cell, no vacuum slab gap."""
    if atoms.cell.volume < 1e-6 or not all(atoms.get_pbc()):
        return False
    return atoms.info.get("provenance", {}).get("type") == "mof"


def free_volume_A3(host: Atoms, clearance: float = 2.0,
                   spacing: float = 0.5) -> float:
    """Volume of the cell where a guest ATOM centre may sit.

    A framework fills most of its own cell, so the cell volume wildly
    overcounts what a guest can use. This grids the cell and keeps points
    further than `clearance` from every framework atom centre, counting
    periodic images so pore space at the borders is measured against the wall
    next door rather than empty space.

    `clearance` is packmol's tolerance (a centre-to-centre distance), so this
    measures the same space packmol will actually pack into. It is a packing
    number, not the adsorption pore volume quoted for a MOF in the literature
    (those use a probe rolled over the vdW surface, a different definition).
    """
    from scipy.spatial import cKDTree

    cell = np.array(host.cell, dtype=float)
    ns = [max(2, int(round(L / spacing))) for L in host.cell.lengths()]
    grid = np.stack(np.meshgrid(*[np.linspace(0, 1, n, endpoint=False)
                                  for n in ns], indexing="ij"), axis=-1)
    points = grid.reshape(-1, 3) @ cell

    pos = np.vstack([host.get_positions() + np.array([i, j, k]) @ cell
                     for i in (-1, 0, 1) for j in (-1, 0, 1)
                     for k in (-1, 0, 1)])
    free = cKDTree(pos).query(points, k=1)[0] > clearance
    return float(free.mean() * host.cell.volume)


def _image_shell(host: Atoms, margin: float) -> Atoms:
    """Framework atoms of the neighbouring cells lying within `margin` of this
    one. packmol has no periodic boundaries, so without these a guest could be
    packed hard against a face and land on top of the wall next door."""
    cell = np.array(host.cell, dtype=float)
    L = np.diag(cell)
    pos, sym = host.get_positions(), np.array(host.get_chemical_symbols())
    keep_p, keep_s = [], []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                if (i, j, k) == (0, 0, 0):
                    continue
                p = pos + np.array([i, j, k]) * L
                m = np.all((p > -margin) & (p < L + margin), axis=1)
                if m.any():
                    keep_p.append(p[m])
                    keep_s.append(sym[m])
    if not keep_p:
        return Atoms()
    return Atoms(symbols=list(np.concatenate(keep_s)),
                 positions=np.vstack(keep_p))


def guest_extents_A(guest: Atoms) -> np.ndarray:
    """The guest's physical size along its own principal axes, vdW included.

    Sorted ascending, so [0] is the thinnest way through a window and [2] is
    the longest span that has to fit in a cage.
    """
    from ase.data import vdw_radii, atomic_numbers

    p = guest.get_positions() - guest.get_positions().mean(axis=0)
    if len(guest) == 1:
        r = _vdw_radius(guest.get_chemical_symbols()[0])
        return np.array([2 * r] * 3)
    # full_matrices keeps all THREE axes: a diatomic spans only two, and a
    # planar ring only two, but they are still 3D objects once vdW is added
    axes = np.linalg.svd(p, full_matrices=True)[2]
    proj = p @ axes.T
    r = np.array([_vdw_radius(s) for s in guest.get_chemical_symbols()])
    ext = (proj + r[:, None]).max(axis=0) - (proj - r[:, None]).min(axis=0)
    return np.sort(ext)


def _vdw_radius(symbol: str) -> float:
    from ase.data import vdw_radii, atomic_numbers
    r = vdw_radii[atomic_numbers[symbol]]
    return float(r) if not np.isnan(r) else 1.7


def _check_guest_fits(host: Atoms, guest: Atoms, name: str) -> None:
    """Refuse a guest that cannot be in the cages at any count.

    The framework's own catalog entry already knows its cage size, so say so
    up front: packmol would otherwise grind through a long search and then
    advise trying fewer, which is nonsense when the honest answer is never.

    Only the CAGE is checked, deliberately. The window is a kinetic question,
    not a geometric one: ZIF-8's linkers swing open, so molecules well past
    its 3.4 A crystallographic aperture (methanol, benzene, hexane) do adsorb
    in real samples. Refusing or warning on window size would state a physical
    claim this agent cannot back with geometry.
    """
    prov = host.info.get("provenance", {})
    cage = prov.get("pore_diameter_A")
    if not cage:
        return
    d = guest_extents_A(guest)
    if d[2] <= cage:
        return
    label = str(prov.get("material", "the framework")).split(" (")[0]
    raise ValueError(
        f"{name or 'that guest'} is {d[2]:.1f} A along its longest axis and "
        f"the cages of {label} are about {cage} A across, so it does not fit "
        "inside the framework at any count, and packing fewer will not help. "
        f"{label} holds small guests: water, CO2, methanol, ethanol, benzene. "
        "A molecule this size needs a framework with larger pores.")


def fill_pores(host: Atoms, guest: Atoms, n: int | None = None,
               density: float | None = None, tolerance: float = 2.0,
               workdir: str | Path = "data/work/pores",
               timeout: int = 900) -> Atoms:
    """Load `n` copies of `guest` into the pores of a periodic framework.

    The framework is fixed and packmol keeps every guest atom `tolerance` away
    from it, so the guests end up in the pore network by construction — no
    pore-finding needed. Guests are packed into the whole cell (minus half a
    tolerance per side, so periodic images meet at exactly the contact
    distance) and the neighbouring cells' walls are shown to packmol as fixed
    atoms, then dropped again.

    n=None loads ~60% of the guest liquid's reference density into the free
    volume — a showcase load, not an equilibrated or experimental uptake.
    """
    import subprocess
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")

    cell = np.array(host.cell, dtype=float)
    if host.cell.volume < 1e-6:
        raise ValueError("the host has no cell, so it has no pores to fill. "
                         "fill_pores needs a periodic framework.")
    if np.abs(cell - np.diag(np.diag(cell))).max() > 1e-6:
        raise ValueError(
            "fill_pores currently handles rectangular cells only (the bundled "
            "MOFs are cubic); this host's cell is not orthogonal.")

    host = host.copy()
    host.wrap()
    L = np.diag(cell)
    m = tolerance / 2.0
    free = free_volume_A3(host, clearance=tolerance)
    if free < 50.0:
        raise ValueError(
            f"the framework has only {free:.0f} A^3 of free volume at a "
            f"{tolerance} A contact distance, too little to pack into. Its "
            "pores may be too narrow for this guest.")

    name = str(guest.info.get("provenance", {}).get("query", "")).lower()
    _check_guest_fits(host, guest, name)
    if n is None:
        rho = density or SOLVENT_DENSITY.get(name, 0.8)
        molar = float(guest.get_masses().sum())
        n = max(1, int(0.6 * rho * (free * 1e-24) * _NA / molar))
        n = min(n, max(1, 8000 // max(1, len(guest))))
    n = int(n)
    if n * len(guest) > 10000:
        raise RuntimeError(
            f"that would pack {n * len(guest)} guest atoms — beyond the "
            "current packing limit (~10,000). Use a smaller count, or load a "
            "single cell and replicate it.")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    shell = _image_shell(host, tolerance)
    walls = host + shell
    host_xyz, guest_xyz = work / "framework.xyz", work / "guest.xyz"
    out_xyz = work / "loaded.xyz"
    write(str(host_xyz), walls, format="xyz")
    write(str(guest_xyz), _centered(guest), format="xyz")
    inp = work / "fill.inp"
    inp.write_text(
        f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
        f"structure {host_xyz.name}\n  number 1\n  fixed 0. 0. 0. 0. 0. 0.\n"
        f"end structure\n\n"
        f"structure {guest_xyz.name}\n  number {n}\n"
        f"  inside box {m:.3f} {m:.3f} {m:.3f} "
        f"{L[0] - m:.3f} {L[1] - m:.3f} {L[2] - m:.3f}\nend structure\n")
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        why = (f"A smaller count will fit." if n > 1 else
               f"Even a single one does not fit: {name or 'the guest'} is "
               f"{guest_extents_A(guest)[2]:.1f} A across and these pores are "
               "too tight for it.")
        raise RuntimeError(
            f"packmol could not fit {n} {name or 'guest'} into the pores. "
            f"{why}\n{proc.stdout[-800:]}")

    packed = read(str(out_xyz))
    keep = list(range(len(host))) + list(range(len(walls), len(packed)))
    loaded = packed[keep]                     # drop the borrowed image walls
    loaded.set_cell(cell)
    loaded.set_pbc(True)
    loaded.info["packmol_inp"] = inp.read_text()
    loaded.info["solute_atoms"] = len(host)   # framework renders solid
    prov = host.info.get("provenance", {})
    loaded.info["provenance"] = prov
    loaded.info["assembly"] = {
        "relation": "inside", "mode": "fill_pores", "n_guests": n,
        "host_atoms": len(host), "guest_atoms": len(guest),
        "free_volume_A3": round(free, 1),
        "guests_per_cell": round(n / max(1, np.prod(prov.get("repeat", [1]))), 2),
        "window_A": prov.get("window_A"),
        "guest_size_A": round(float(guest_extents_A(guest)[2]), 1),
    }
    return loaded


def _is_slab(atoms: Atoms) -> bool:
    """Periodic cell with vacuum along z — a surface slab, not a free particle."""
    if atoms.cell.volume < 1e-6:
        return False
    z = atoms.get_positions()[:, 2]
    return atoms.cell.lengths()[2] - (z.max() - z.min()) > 6.0


def _liquid_volume_A3(molecule: Atoms, n: int) -> float:
    """Approximate liquid-state volume of n copies (from the reference density)."""
    name = str(molecule.info.get("provenance", {}).get("query", "")).lower()
    rho = SOLVENT_DENSITY.get(name, 0.9)
    return n * float(molecule.get_masses().sum()) / (rho * _NA) * 1e24


def _film_footprint(slab: Atoms, tolerance: float):
    """In-plane packmol constraints that make a film span the WHOLE periodic
    cell (not the atoms' bounding box, which leaves a vacuum seam at the
    borders). Returns (extra_lines, (x0, y0, x1, y1), area_A2). Rectangular
    cells use the box directly; sheared cells (hexagonal terminations) add
    four half plane constraints bounding the cell parallelogram, with half a
    tolerance margin per side so periodic images meet at the packmol contact
    distance."""
    p = slab.get_positions()
    lo, hi = p.min(axis=0), p.max(axis=0)
    cellm = np.array(slab.cell, dtype=float)
    a, b = cellm[0][:2], cellm[1][:2]
    m = tolerance / 2.0
    if abs(a[1]) < 1e-3 and abs(b[0]) < 1e-3:      # rectangular in-plane cell
        x0, x1 = min(lo[0], 0.0) + m, max(hi[0], a[0]) - m
        y0, y1 = min(lo[1], 0.0) + m, max(hi[1], b[1]) - m
        return "", (x0, y0, x1, y1), max((x1 - x0) * (y1 - y0), 1.0)
    lines = []
    for v, w in ((a, b), (b, a)):                  # edges run along w
        nvec = np.array([w[1], -w[0]])
        nvec = nvec / np.linalg.norm(nvec)
        if float(nvec @ v) < 0:
            nvec = -nvec
        d = float(nvec @ v)
        lines.append(f"  over plane {nvec[0]:.6f} {nvec[1]:.6f} 0. {m:.3f}")
        lines.append(f"  below plane {nvec[0]:.6f} {nvec[1]:.6f} 0. {d - m:.3f}")
    x0 = min(lo[0], 0.0, a[0], b[0], a[0] + b[0]) - 1.0
    y0 = min(lo[1], 0.0, a[1], b[1], a[1] + b[1]) - 1.0
    x1 = max(hi[0], 0.0, a[0], b[0], a[0] + b[0]) + 1.0
    y1 = max(hi[1], 0.0, a[1], b[1], a[1] + b[1]) + 1.0
    area = abs(a[0] * b[1] - a[1] * b[0])
    return "\n".join(lines), (x0, y0, x1, y1), max(area, 1.0)


def _coat_slab(slab: Atoms, molecules: list, ns: list,
               clearance: float = 3.5, tolerance: float = 2.0,
               workdir: str | Path = "data/work/coat", timeout: int = 300) -> Atoms:
    """Coat a SLAB: fill the vacuum of its simulation cell with the molecules —
    ALL guest species packed in ONE packmol run (slab fixed), one shared cell.

    ns entries of None fill the leftover volume at ~liquid density (capped)."""
    import subprocess
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")

    p = slab.get_positions()
    lo, hi = p.min(axis=0), p.max(axis=0)
    Lz = float(slab.cell.lengths()[2])
    z0, z1 = hi[2] + clearance, Lz - clearance
    # the film fills the WHOLE in-plane periodic cell (rectangle or sheared
    # parallelogram), never the atoms' bounding rectangle
    extra_lines, (x0, y0, x1, y1), area = _film_footprint(slab, tolerance)
    # if the EXPLICIT counts need more room, grow the cell along z — the in-plane
    # cell (x, y) is fixed by the slab supercell and never changes
    fixed_need = sum(_liquid_volume_A3(m_, int(n))
                     for m_, n in zip(molecules, ns) if n) / 0.75
    z1 = max(z1, z0 + fixed_need / area)
    new_Lz = max(Lz, z1 + clearance)
    if z1 - z0 < 3.0:
        raise RuntimeError("no vacuum above the surface to fill — increase the "
                           "slab's vacuum parameter")
    box_lo = (x0, y0, z0)
    box_hi = (x1, y1, z1)
    volume_A3 = area * (z1 - z0)

    # fixed counts reserve their liquid volume; auto entries share the rest
    fixed_vol = sum(_liquid_volume_A3(m, int(n))
                    for m, n in zip(molecules, ns) if n)
    autos = [i for i, n in enumerate(ns) if not n]
    free = max(volume_A3 - fixed_vol, 0.0)
    counts = []
    for i, (m, n) in enumerate(zip(molecules, ns)):
        if n:
            counts.append(int(n))
        else:
            name = str(m.info.get("provenance", {}).get("query", "")).lower()
            rho = SOLVENT_DENSITY.get(name, 0.9)
            molar = float(m.get_masses().sum())
            share = free / max(len(autos), 1)
            c = max(1, int(0.7 * rho * (share * 1e-24) * _NA / molar))
            counts.append(min(c, max(1, 8000 // max(1, len(m)))))

    total_guest_atoms = sum(c * len(m) for m, c in zip(molecules, counts))
    if total_guest_atoms > 10000:
        raise RuntimeError(
            f"that would pack {total_guest_atoms} guest atoms — beyond the current "
            "packing limit (~10,000; packmol converges too slowly for one huge "
            "fill). Suggestion: build a SMALLER cell first (fewer molecules, or a "
            "smaller NxM slab supercell) and replicate it periodically to reach "
            "the target size. Support for very large fills (sliced packing) is "
            "planned.")

    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    slab_xyz, out_xyz = work / "slab.xyz", work / "coated.xyz"
    write(str(slab_xyz), slab, format="xyz")
    blocks = [f"structure {slab_xyz.name}\n  number 1\n"
              "  fixed 0. 0. 0. 0. 0. 0.\nend structure\n"]
    box_line = "  inside box " + " ".join(f"{v:.3f}" for v in (*box_lo, *box_hi))
    if extra_lines:
        box_line += "\n" + extra_lines
    for i, (m, c) in enumerate(zip(molecules, counts)):
        mol_xyz = work / f"mol{i}.xyz"
        write(str(mol_xyz), _centered(m), format="xyz")
        blocks.append(f"structure {mol_xyz.name}\n  number {c}\n{box_line}\n"
                      "end structure\n")
    inp = work / "coat.inp"
    inp.write_text(f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
                   + "\n".join(blocks))
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol coat (slab fill) failed:\n{proc.stdout[-1200:]}")

    coated = read(str(out_xyz))
    coated.info["packmol_inp"] = inp.read_text()
    cell = np.array(slab.cell)                 # ONE simulation cell for everything;
    if new_Lz > Lz + 1e-6:                     # grown along z only if needed
        cell[2] = [0.0, 0.0, new_Lz]
    coated.set_cell(cell)
    coated.set_pbc(slab.pbc)
    coated.info["assembly"] = {"relation": "coated_by", "mode": "fill_cell",
                               "n_guests": counts, "host_atoms": len(slab),
                               "guest_atoms": [len(m) for m in molecules],
                               "filled_volume_A3": round(volume_A3, 1),
                               "cell_z_extended_A": round(new_Lz - Lz, 1)}
    if slab.info.get("provenance"):        # keep the slab's identity (miller etc.)
        coated.info["provenance"] = dict(slab.info["provenance"])
    return coated


def coat_layers(host: Atoms, guests: list, ns: list | None = None,
                workdir: str | Path = "data/work/coat") -> Atoms:
    """Coat `host` with SEVERAL guest species at once.

    Slab hosts: one packmol run fills the cell's vacuum with all species together
    (n=None entries share the leftover volume at ~liquid density).
    Particle hosts: sequential shell coats.
    """
    ns = list(ns) if ns is not None else [None] * len(guests)
    if _is_slab(host):
        return _coat_slab(host, list(guests), ns, workdir=workdir)
    out = host
    for g, n in zip(guests, ns):
        out = coat(out, g, n=n, workdir=workdir)
    return out


def coat(core: Atoms, molecule: Atoms, n: int | None = None,
         shell_thickness: float = 8.0,
         workdir: str | Path = "data/work/coat") -> Atoms:
    """Coat `core` with copies of `molecule`.

    Slab hosts: the simulation cell's vacuum is FILLED with the molecules (packmol,
    n=None -> liquid-density fill) — both end up in the same cell.
    Particle hosts: `n` copies (default 30) packed as a spherical shell, then
    assembled via MOLTEMPLATE — the molecule is defined once as a reusable .lt
    object and instantiated n times with the Kabsch-recovered .rot().move()
    transforms; the emitted .lt passes Gate 1 (KG validation) and Gate 2
    (moltemplate.sh) before the geometry is read back.
    """
    if _is_slab(core):
        return _coat_slab(core, [molecule], [n], workdir=workdir)
    n = int(n or 30)
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    packed, meta = pack_shell(core, molecule, int(n),
                              shell_thickness=shell_thickness, workdir=work)
    emit_assembly(core, molecule, packed, int(n), work)

    kg = MoltemplateKG()
    lt_files = {f: (work / f).read_text() for f in ("system.lt", "core.lt", "mol.lt")}
    for fname, text in lt_files.items():
        rep = kg.validate_lt(text)
        if not rep.passed:
            raise RuntimeError(f"GATE 1 FAILED in {fname}:\n{rep.summary()}")

    data_file = run_moltemplate(work / "system.lt", work)
    elements, positions = parse_lammps_data(data_file)
    coated = Atoms(elements, positions=positions)
    coated.info["packmol_inp"] = meta.get("packmol_inp", "")
    coated.info["assembly"] = {"relation": "coated_by", "mode": "shell",
                               "n_guests": int(n), "host_atoms": len(core),
                               "expected_atoms": meta["expected_atoms"],
                               "shell": [meta["shell_inner"], meta["shell_outer"]]}
    coated.info["lt_files"] = lt_files
    return coated


def sandwich(slab: Atoms, molecule: Atoms, n: int | None = None,
             gap: float | None = None, top: Atoms | None = None,
             clearance: float = 3.5, tolerance: float = 2.0,
             workdir: str | Path = "data/work/between",
             timeout: int = 300) -> Atoms:
    """Confine a liquid film of `molecule` BETWEEN two slabs.

    A film of thickness `gap` is packed above `slab`'s top surface (packmol,
    slab fixed), then the `top` slab (a copy of `slab` if not given — the
    homo-sandwich) is stacked on top — one periodic cell: bottom slab + film
    + top slab + the top slab's original vacuum. `n` fixes the number of
    molecules (the gap grows to hold them at liquid density); without `n` the
    default 12 Å gap is filled at ~liquid density.

    A different `top` slab must share the bottom slab's in-plane cell within
    5% — otherwise pick matching NxM supercells (repeat) first.
    """
    import subprocess
    if not _is_slab(slab):
        what = ("a bulk crystal (fully periodic, no vacuum)"
                if slab.cell.volume > 1e-6
                else "a free particle/molecule (no periodic cell)")
        raise ValueError(
            f"sandwich() needs a surface SLAB host, but the host "
            f"({slab.get_chemical_formula()}) is {what}. Ask for a surface "
            "of the material instead — e.g. 'GaAs 110 slab' — so there is a "
            "surface to confine the film against.")
    strain = 0.0
    if top is not None:
        la, lb = slab.cell.lengths()[:2], top.cell.lengths()[:2]

        def _ang(c):
            return float(np.degrees(np.arccos(
                np.dot(c[0], c[1]) / (np.linalg.norm(c[0]) * np.linalg.norm(c[1])))))
        d_ang = abs(_ang(slab.cell[:]) - _ang(top.cell[:]))
        if d_ang > 8.0:
            raise ValueError(
                f"the two surface cells have different SHAPES: in-plane angles "
                f"{_ang(slab.cell[:]):.0f}° (bottom) vs {_ang(top.cell[:]):.0f}° "
                "(top) — no supercell repeat can reconcile e.g. a hexagonal "
                "termination with a rectangular one. Pick terminations with "
                "matching cell shapes (e.g. magnetite (001) with rutile (110) "
                "or (001), both rectangular) and rebuild.")
        strain = max(abs(x - y) / max(x, y) for x, y in zip(la, lb))
        if strain > 0.12:
            raise ValueError(
                f"in-plane cells too different: bottom {la[0]:.2f}x{la[1]:.2f} Å "
                f"vs top {lb[0]:.2f}x{lb[1]:.2f} Å (~{strain:.0%} strain, cap "
                "12%). Choose NxM supercells (the 'repeat' parameter) that "
                "bring the two slabs closer in in-plane size, then rebuild.")
        if strain > 1e-6:            # strain the TOP film onto the bottom cell
            top = top.copy()         # (standard epitaxial-interface practice)
            newc = np.array(top.cell)
            newc[0], newc[1] = np.array(slab.cell[0]), np.array(slab.cell[1])
            top.set_cell(newc, scale_atoms=True)
    packmol = find_packmol()
    if packmol is None:
        raise RuntimeError("packmol not found on PATH")

    p = slab.get_positions()
    lo, hi = p.min(axis=0), p.max(axis=0)
    thickness = float(hi[2] - lo[2])
    # film spans the WHOLE in-plane cell (rectangle or sheared parallelogram)
    film_lines, (bx0, by0, bx1, by1), area = _film_footprint(slab, tolerance)

    # packmol keeps the film `tolerance` away from the FIXED bottom slab, but
    # the top slab is stacked afterwards — its margin must be built into the
    # packing box or the film visually overlaps the top surface
    top_margin = max(clearance, tolerance)
    if gap is None:
        gap = (_liquid_volume_A3(molecule, int(n)) / 0.75 / area
               + clearance + top_margin) if n else 12.0
    gap = max(float(gap), clearance + top_margin + 2.0)
    if n is None:
        name = str(molecule.info.get("provenance", {}).get("query", "")).lower()
        rho = SOLVENT_DENSITY.get(name, 0.9)
        molar = float(molecule.get_masses().sum())
        vol = area * (gap - 2 * clearance)
        n = max(1, int(0.7 * rho * (vol * 1e-24) * _NA / molar))
        n = min(n, max(1, 8000 // max(1, len(molecule))))
    n = int(n)
    if n * len(molecule) > 10000:
        raise RuntimeError(
            f"that would pack {n * len(molecule)} film atoms — beyond the "
            "current packing limit (~10,000). Use a smaller slab supercell or "
            "an explicit smaller count.")

    # position the top slab FIRST and include it in the packmol run as a
    # second fixed wall — corrugated surfaces (protruding atom rows) make a
    # purely geometric box margin insufficient on the top side
    top_slab = (top if top is not None else slab).copy()
    tp = top_slab.get_positions()
    top_thickness = float(tp[:, 2].max() - tp[:, 2].min())
    top_vacuum = max(float(top_slab.cell.lengths()[2]) - tp[:, 2].max(),
                     clearance)
    top_slab.translate([0.0, 0.0, (hi[2] + gap) - tp[:, 2].min()])

    z0, z1 = hi[2] + clearance, hi[2] + gap - clearance
    work = Path(workdir)
    work.mkdir(parents=True, exist_ok=True)
    slab_xyz, top_xyz, mol_xyz, out_xyz = (work / "slab.xyz", work / "top.xyz",
                                           work / "mol.xyz", work / "film.xyz")
    write(str(slab_xyz), slab, format="xyz")
    write(str(top_xyz), top_slab, format="xyz")
    write(str(mol_xyz), _centered(molecule), format="xyz")
    inp = work / "between.inp"
    inp.write_text(
        f"tolerance {tolerance}\nfiletype xyz\noutput {out_xyz.name}\n\n"
        f"structure {slab_xyz.name}\n  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"structure {top_xyz.name}\n  number 1\n"
        "  fixed 0. 0. 0. 0. 0. 0.\nend structure\n\n"
        f"structure {mol_xyz.name}\n  number {n}\n"
        f"  inside box {bx0:.3f} {by0:.3f} {z0:.3f} "
        f"{bx1:.3f} {by1:.3f} {z1:.3f}\n"
        + (film_lines + "\n" if film_lines else "")
        + "end structure\n")
    proc = subprocess.run([packmol], stdin=inp.open(), cwd=work,
                          capture_output=True, text=True, timeout=timeout)
    if not out_xyz.exists() or "Success" not in proc.stdout:
        raise RuntimeError(f"packmol sandwich fill failed:\n{proc.stdout[-1200:]}")

    combined = read(str(out_xyz))                  # bottom + top + film
    cell = np.array(slab.cell)
    cell[2] = [0.0, 0.0, hi[2] + gap + top_thickness + top_vacuum]
    combined.set_cell(cell)
    combined.set_pbc(True)
    combined.info["packmol_inp"] = inp.read_text()
    combined.info["assembly"] = {"relation": "between", "mode": "sandwich",
                                 "n_guests": n, "host_atoms": len(slab),
                                 "gap_A": round(gap, 2),
                                 "film_atoms": n * len(molecule),
                                 "top_strain_pct": round(strain * 100, 2)}
    if slab.info.get("provenance"):
        combined.info["provenance"] = dict(slab.info["provenance"])
    return combined


def place_on(host: Atoms, guest: Atoms, n: int = 1, height: float = 3.5) -> Atoms:
    """Place `n` copies of `guest` on the top surface of `host` (grid in-plane)."""
    host = host.copy()
    hp = host.get_positions()
    top = hp[:, 2].max()
    lo, hi = hp[:, :2].min(axis=0), hp[:, :2].max(axis=0)
    side = int(np.ceil(np.sqrt(n)))
    span = hi - lo
    combined = host
    for k in range(int(n)):
        i, j = divmod(k, side)
        xy = lo + span * ((np.array([i, j]) + 0.5) / side)
        g = guest.copy()
        gp = g.get_positions()
        gp -= gp.mean(axis=0)
        gp[:, 2] -= gp[:, 2].min()                     # rest the guest on its base
        g.set_positions(gp + [xy[0], xy[1], top + height])
        combined = combined + g
    combined.info["assembly"] = {"relation": "on", "n_guests": int(n), "height": height}
    return combined


# ------------------- showcase snippet templates (per relation) -----------------

def combine_template_multi(relations: list, by_key: dict) -> str:
    """Assembly snippet for MULTIPLE relations. Consecutive coatings of the SAME
    host are packed together in one coat_layers call (they share the free volume);
    other relations apply sequentially to the growing `atoms` structure."""
    steps: list[tuple] = []                 # ("coat_group"|"around_group", host, [rels]) | ("one", rel)
    for rel in relations:
        kind = rel["kind"]
        group = {"coated_by": "coat_group", "around": "around_group"}.get(kind)
        if group and steps and steps[-1][0] == group and steps[-1][1] == rel["host"]:
            steps[-1][2].append(rel)
        elif group:
            steps.append((group, rel["host"], [rel]))
        else:
            steps.append(("one", rel))

    lines: list[str] = []
    imports: list[str] = []
    consumed: set[str] = set()

    def add(code: str) -> None:
        for ln in code.splitlines():
            if ln.startswith("from "):
                if ln not in imports:
                    imports.append(ln)
            else:
                lines.append(ln)

    for step in steps:
        if step[0] == "coat_group" and len(step[2]) > 1:
            host_key, rels = step[1], step[2]
            host = "atoms" if host_key in consumed else host_key
            guests = ", ".join(r["guest"] for r in rels)
            ns = ", ".join(str((r.get("params") or {}).get("count") or None)
                           for r in rels)
            add("from mtagent.assemble import coat_layers\n"
                f"atoms = coat_layers({host}, [{guests}], ns=[{ns}])")
            consumed.add(host_key)
        elif step[0] == "around_group" and len(step[2]) > 1 and any(
                by_key[r["guest"]]["builder"] == "solvent_box" for r in step[2]):
            # several species around one host: extra guests DISSOLVE into the
            # solvent box, then ONE carve-and-insert solvation
            host_key, rels = step[1], step[2]
            host = "atoms" if host_key in consumed else host_key
            box_r = next(r for r in rels
                         if by_key[r["guest"]]["builder"] == "solvent_box")
            code = [f"box = {box_r['guest']}"]
            for r in rels:
                if r is box_r:
                    continue
                cnt = (r.get("params") or {}).get("count") or 10
                code.append(f"box = add_to_box(box, {r['guest']}, n={int(cnt)})")
            code.append(f"atoms = solvate({host}, box)")
            add("from mtagent.solvent import add_to_box, solvate\n"
                + "\n".join(code))
            consumed.add(host_key)
        else:
            rels_g = step[2] if step[0] in ("coat_group", "around_group") else [step[1]]
            for rel in rels_g:
                host = "atoms" if rel["host"] in consumed else rel["host"]
                add(combine_template(rel["kind"], host, rel["guest"],
                                     by_key[rel["guest"]]["builder"],
                                     rel.get("params") or {},
                                     host_builder=(by_key.get(rel["host"]) or {})
                                     .get("builder")))
                consumed.add(rel["host"])
    return "\n".join(imports + lines)


def combine_template(kind: str, host_key: str, guest_key: str,
                     guest_builder: str, params: dict,
                     host_builder: str | None = None) -> str:
    """Canonical assembly snippet. Constituent keys are variables in the exec namespace."""
    n = params.get("count") or params.get("n")
    try:
        n = int(n)                     # LLM may answer "auto"/None -> auto-fill
    except (TypeError, ValueError):
        n = None
    if kind == "inside":
        n_arg = f", n={int(n)}" if n else ""
        if host_builder == "mof":      # pores, not a bounding box
            return ("from mtagent.assemble import fill_pores\n"
                    f"atoms = fill_pores({host_key}, {guest_key}{n_arg})")
        return ("from mtagent.assemble import fill_inside\n"
                f"atoms = fill_inside({host_key}, {guest_key}{n_arg})")
    if kind == "around":
        if guest_builder == "solvent_box":
            return ("from mtagent.solvent import solvate\n"
                    f"atoms = solvate({host_key}, {guest_key})")
        size = float(params.get("box_size") or 40.0)
        n_arg = f", n={n}" if n else ""
        return ("from mtagent.solvent import build_solvent_box, solvate\n"
                f"box = build_solvent_box({guest_key}, box_size={size:g}{n_arg})\n"
                f"atoms = solvate({host_key}, box)")
    if kind == "coated_by":
        n_arg = f", n={int(n)}" if n else ""     # no n: slab -> fill cell, NP -> 30
        return ("from mtagent.assemble import coat\n"
                f"atoms = coat({host_key}, {guest_key}{n_arg})")
    if kind == "on":
        return ("from mtagent.assemble import place_on\n"
                f"atoms = place_on({host_key}, {guest_key}, n={int(n or 1)})")
    if kind == "between":
        n_arg = f", n={int(n)}" if n else ""
        top_arg = (f", top={params['second_host']}"
                   if params.get("second_host") else "")
        return ("from mtagent.assemble import sandwich\n"
                f"atoms = sandwich({host_key}, {guest_key}{n_arg}{top_arg})")
    raise ValueError(f"unknown relation '{kind}' (have: {sorted(RELATIONS)})")
