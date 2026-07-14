"""GROUND — KG retrieval + gap check. Runs BEFORE any LLM generation, with no LLM.

For each parsed constituent this stage retrieves the evidence the proposal step needs
(real introspected signatures from the API KG, Moltemplate KG constraints when the
build assembles via .lt) and computes what is genuinely MISSING from the user's query:
required builder parameters without a value, and the relation between constituents.
Only those gaps become clarifier questions — never a scripted question list.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

from .ase_kg import ASEKnowledgeGraph, MODULES, MTAGENT_MODULES
from .ase_validate import ASEValidator, referenced_entries
from .assemble import RELATIONS, combine_template, combine_template_multi, relation_catalog
from .cluster import cluster_extent
from .kg import MoltemplateKG
from .registry import BUILDERS, _is_oxide, normalize_material, slug

# Built once per process: introspection of ase + mtagent, and the manual-derived KG.
_API_KG: ASEKnowledgeGraph | None = None
_MT_KG: MoltemplateKG | None = None
_VALIDATOR: ASEValidator | None = None


def api_kg() -> ASEKnowledgeGraph:
    global _API_KG
    if _API_KG is None:
        _API_KG = ASEKnowledgeGraph(MODULES + MTAGENT_MODULES, prefixes=("ase", "mtagent"))
    return _API_KG


def moltemplate_kg() -> MoltemplateKG:
    global _MT_KG
    if _MT_KG is None:
        _MT_KG = MoltemplateKG()
    return _MT_KG


def validator() -> ASEValidator:
    global _VALIDATOR
    if _VALIDATOR is None:
        _VALIDATOR = ASEValidator(api_kg())
    return _VALIDATOR


# ------------------------------- evidence packs --------------------------------

@dataclass
class Evidence:
    target: str                       # constituent key, or "showcase"
    template: str                     # canonical registry snippet
    signatures: list[str] = field(default_factory=list)   # real introspected signatures
    notes: list[str] = field(default_factory=list)        # Moltemplate KG constraints
    variables: list[str] = field(default_factory=list)    # predefined vars in exec ns

    def as_prompt(self) -> str:
        parts = ["Signatures of the ONLY functions you may call (introspected, exact):"]
        parts += [f"  {s}" for s in self.signatures]
        if self.variables:
            parts += ["Predefined variables (already-built ase.Atoms objects): "
                      + ", ".join(self.variables)]
        if self.notes:
            parts += ["Moltemplate constraints:"] + [f"  {n}" for n in self.notes]
        parts += ["Canonical snippet (KG-validated reference):", self.template]
        return "\n".join(parts)


def _sig_lines(code: str) -> list[str]:
    return [f"{e.qualname}{e.signature}" + (f"  # {e.doc}" if e.doc else "")
            for e in referenced_entries(code, api_kg())]


def _mt_notes(request: str) -> list[str]:
    r = moltemplate_kg().retrieve(request)
    allowed = ", ".join(sorted(r.allowed)) or "(none)"
    return [*r.notes, f"allowed .lt constructs: {allowed}"]


def evidence_for(constituent: dict) -> Evidence:
    """RETRIEVE for one constituent: template + real signatures (+ .lt constraints)."""
    b = BUILDERS[constituent["builder"]]
    spec = b.defaults(constituent.get("spec") or {})
    template = b.template(spec)
    notes = []
    if int(spec.get("n_particles") or 1) > 1:      # cluster assembles via Moltemplate
        notes = _mt_notes("assemble copies of a rigid unit, geometry only")
    return Evidence(constituent["key"], template, _sig_lines(template), notes)


def relations_of(state: dict) -> list:
    """The relations list (accepts the legacy single-relation shape)."""
    rels = state.get("relations")
    if rels is None:
        rels = [state["relation"]] if state.get("relation") else []
    return rels


def evidence_for_relation(state: dict) -> Evidence | None:
    rels = relations_of(state)
    if not rels:
        return None
    by_key = {c["key"]: c for c in state["constituents"]}
    template = combine_template_multi(rels, by_key)
    notes = _mt_notes("molecule shell assembly, geometry only") \
        if any(r["kind"] == "coated_by" for r in rels) else []
    return Evidence("showcase", template, _sig_lines(template), notes,
                    variables=list(by_key))


# --------------------------------- gap check -----------------------------------

@dataclass
class Gap:
    target: str                       # constituent key, or "relation"
    param: str
    question: str


def _by_key(state: dict, key: str) -> dict:
    return next(c for c in state["constituents"] if c["key"] == key)


def gaps(state: dict) -> list[Gap]:
    """What the retrieved builders require that the query didn't determine."""
    out: list[Gap] = []
    for c in state["constituents"]:
        b = BUILDERS.get(c["builder"])
        if b is None:
            continue                   # parse is registry-constrained; tolerate anyway
        spec = c.get("spec") or {}
        for p in b.params:
            if not (p.required and p.ask) or spec.get(p.name) is not None:
                continue
            if p.when is not None and not p.when(spec):
                continue
            out.append(Gap(c["key"], p.name, f"[{c['key']}] {p.ask}"))

    cs = state["constituents"]
    rels = relations_of(state)
    if len(cs) >= 2 and not rels:
        names = " and ".join(c["key"] for c in cs[:2])
        out.append(Gap("relation", "kind",
                       f"How should {names} be combined?\n{relation_catalog()}"))
    for rel in rels:
        if rel["kind"] in ("inside", "coated_by") \
                and "count" not in (rel.get("params") or {}):
            what = "packed inside" if rel["kind"] == "inside" else "coating the surface"
            default = ("auto — fill at liquid density" if rel["kind"] == "inside"
                       else "30")
            out.append(Gap("relation", "count",
                           f"How many {rel['guest']} molecules {what}? "
                           f"(default: {default})"))
    return out


def finalize(state: dict) -> dict:
    """Fill declared defaults + derived values once no gaps remain (deterministic)."""
    st = copy.deepcopy(state)
    key_map = {}
    for c in st["constituents"]:
        key_map[c["key"]] = c["key"] = slug(c["key"])
        b = BUILDERS.get(c["builder"])
        if b is None:
            continue
        c["spec"] = spec = b.defaults(c.get("spec") or {})
        if c["builder"] == "nanoparticle":
            normalize_material(spec)
            if not spec.get("shape"):
                spec["shape"] = "wulff" if _is_oxide(spec) else "sphere"

    rels = relations_of(st)
    for rel in rels:
        rel["host"], rel["guest"] = key_map.get(rel["host"], rel["host"]), \
            key_map.get(rel["guest"], rel["guest"])
        rel.setdefault("params", {})
    st["relations"] = rels
    st["relation"] = rels[0] if rels else None            # compat alias

    # a solvent box (or an 'around' fill) must ENCOMPASS the thing it surrounds
    host_size = _solute_size(st)
    for c in st["constituents"]:
        if c["builder"] == "solvent_box":
            c["spec"]["box_size"] = max(float(c["spec"].get("box_size") or 0.0),
                                        host_size + 20.0)
    by_key = {c["key"]: c.get("builder") for c in st["constituents"]}
    for rel in rels:
        if rel["kind"] == "around":
            if by_key.get(rel["guest"]) == "solvent_box":
                # the guest box carries its own (clamped) size — advertising a
                # box_size here tempts the LLM to pass it into solvate's third
                # positional, which is the clash cutoff (deletes all solvent)
                rel["params"].pop("box_size", None)
            else:
                rel["params"]["box_size"] = max(
                    float(rel["params"].get("box_size") or 0.0), host_size + 20.0)

    # hetero-sandwich: the two slabs' in-plane cells must (nearly) match —
    # pick NxM supercells for both automatically unless the user chose them
    cons_by_key = {c["key"]: c for c in st["constituents"]}
    for rel in rels:
        sh = (rel.get("params") or {}).get("second_host")
        if rel["kind"] != "between" or not sh:
            continue
        c1, c2 = cons_by_key.get(rel["host"]), cons_by_key.get(sh)
        if not (c1 and c2 and c1["builder"] == c2["builder"] == "surface_slab"):
            continue
        if c1["spec"].get("repeat") or c2["spec"].get("repeat"):
            continue                               # user's explicit choice wins
        u1, u2 = _slab_inplane_unit(c1["spec"]), _slab_inplane_unit(c2["spec"])
        if not (u1 and u2):
            continue
        if abs(u1[2] - u2[2]) > 8.0:
            continue    # cell SHAPES differ (e.g. hexagonal vs square) — no
            #             repeat can fix that; sandwich() raises the guiding error
        mx = _match_repeats(u1[0], u2[0])
        my = _match_repeats(u1[1], u2[1])
        if mx and my:
            c1["spec"]["repeat"] = f"{mx[0]}x{my[0]}"
            c2["spec"]["repeat"] = f"{mx[1]}x{my[1]}"
    return st


def _slab_inplane_unit(spec: dict) -> tuple[float, float, float] | None:
    """(lx, ly, angle_deg) of this slab spec's 1x1 surface cell (thin probe)."""
    import numpy as np
    from .execute import run_snippet
    from .registry import BUILDERS
    probe = dict(spec, repeat="1x1",
                 thickness=min(float(spec.get("thickness") or 10.0), 5.0))
    try:
        atoms = run_snippet(BUILDERS["surface_slab"].template(probe))
        lx, ly = atoms.cell.lengths()[:2]
        v0, v1 = np.array(atoms.cell[0]), np.array(atoms.cell[1])
        ang = float(np.degrees(np.arccos(
            np.dot(v0, v1) / (np.linalg.norm(v0) * np.linalg.norm(v1)))))
        return float(lx), float(ly), ang
    except Exception:
        return None


def _match_repeats(u1: float, u2: float, max_rep: int = 8,
                   max_len: float = 45.0) -> tuple[int, int] | None:
    """Repeat counts (n1, n2) with n1*u1 ≈ n2*u2: minimal strain, then minimal
    size, under the sandwich's 12% strain cap and a size budget."""
    best = None
    for n1 in range(1, max_rep + 1):
        for n2 in range(1, max_rep + 1):
            L1, L2 = u1 * n1, u2 * n2
            if max(L1, L2) > max_len:
                continue
            strain = abs(L1 - L2) / max(L1, L2)
            if strain > 0.12:
                continue
            score = (round(strain, 3), max(L1, L2))
            if best is None or score < best[0]:
                best = (score, n1, n2)
    return (best[1], best[2]) if best else None


def _solute_size(state: dict) -> float:
    """Pre-build size estimate of the host/solute, from its spec (Å)."""
    rels = relations_of(state)
    host_key = rels[0]["host"] if rels else None
    for c in state["constituents"]:
        if host_key and c["key"] != host_key:
            continue
        s = c.get("spec") or {}
        if c["builder"] == "nanoparticle":
            d = float(s.get("diameter") or 40.0)
            n = int(s.get("n_particles") or 1)
            return cluster_extent(d, n, float(s.get("gap") or 10.0)) if n > 1 else d
        if c["builder"] == "nanotube":
            return float(int(s.get("length") or 10)) * 2.46
        if c["builder"] == "surface_slab":
            return 30.0
    return 40.0
