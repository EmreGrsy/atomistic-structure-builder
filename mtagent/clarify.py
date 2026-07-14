"""PARSE — the LLM's two jobs in the chat: parse the query, and parse each answer.

`parse_query` turns the user's request into typed constituents + the relation between
them, constrained to the live builder registry (the catalog in the prompt is DERIVED
from mtagent.registry — never hand-written). `apply_answer` merges the user's reply to
a clarifier question into the spec. Which questions get asked is NOT decided here:
ground.gaps() computes that from the registry schemas (KG-first, ask only what's
missing). Without an OpenAI key a small keyword parser keeps the demo path alive.
"""
from __future__ import annotations

import json
import re

from .assemble import RELATIONS, relation_catalog
from .ground import Gap
from .llm import DEFAULT_MODEL, chat_json, have_openai_key
from .registry import BUILDERS, catalog, slug

_STATE_SHAPE = """Return ONLY JSON of this exact shape:
{"constituents": [{"key": "<short_snake_case>", "builder": "<registry name>",
                   "spec": {<only fields the text explicitly determines>}}],
 "relations": [{"kind": "<relation>", "host": "<key>", "guest": "<key>",
                "params": {}}, ...] (empty list if none stated),
 "summary": "<one short sentence restating the structure>"}
Each guest gets its OWN entry in "relations" (e.g. a slab coated with oleic acid AND
30 water molecules = two entries with the same host). Adding another molecule to an
existing system = APPEND a relation entry, never replace the existing ones."""


def _parse_system() -> str:
    return f"""You parse a user's molecular-structure request into typed constituents.

Builders (use these EXACT names; do not invent others):
{catalog()}

Relations between two constituents (host = the larger/framework structure):
{relation_catalog()}

Rules: 1 nm = 10 Angstrom; diameter = 2 * radius. Set only spec fields the text
explicitly determines — leave everything else out (the system asks the user about
genuinely missing fields itself). A structure described as in / inside / immersed in
a LIQUID (water, ethanol, ...) is SOLVATED: the liquid is a solvent_box guest with
relation kind "around" (host = the structure) — the "inside" kind is ONLY for guests
contained within a hollow host (e.g. molecules inside a nanotube). A surface_slab
coated with / covered by / in a liquid: the liquid is a plain molecule guest with
relation "coated_by" (its cell gets filled). Set solvent_box "n" ONLY when the user
explicitly states a number of solvent molecules — never invent one. For a nanoparticle
give "material" as a plain mineral/metal name (e.g. "magnetite", "gold") — no formulas
or qualifiers. Set "relation" only if the text states or clearly implies it, else null.
{_STATE_SHAPE}"""


def _apply_system() -> str:
    return f"""You update a molecular build spec from the user's chat message.

Builders and their spec fields:
{catalog()}

Relations between two constituents:
{relation_catalog()}

Merge ONLY what the user's message determines into the JSON you are given — it may
adjust any number of fields at once (e.g. "(8,8), 20 cells, and put it in water"),
answer an open question, add/remove a constituent, or set how constituents combine
(relation.kind, and relation.params.count for how many guest copies). Counts: for a
solvent_box constituent a NUMBER of solvent molecules (e.g. "5 water molecules")
goes into that constituent's spec field "n"; for a molecule guest of an
inside/coated_by relation it goes to relation.params.count — never elsewhere.
A count around a SOLID host keeps relation kind "around" (only hollow hosts contain
guests). KEEP every constituent "key" exactly as given (never rename keys) and keep
relation.host/guest pointing at those keys. If the user accepts a stated default,
set that default value. Do not change anything the user didn't address.
Rules: 1 nm = 10 Angstrom; diameter = 2 * radius.
{_STATE_SHAPE}"""


def _sanitize_relation(rel, cs: list, keys: list) -> dict | None:
    """Validate/normalize ONE relation against the live constituents."""
    if not isinstance(rel, dict) or rel.get("kind") not in RELATIONS or len(cs) < 2:
        return None
    # LLM sometimes renames keys mid-chat — remap to live keys, don't drop
    for field_ in ("host", "guest"):
        k = slug(rel.get(field_) or "")
        if k not in keys:
            match = next((x for x in keys if k and (k in x or x in k)), None)
            rel[field_] = match if match else \
                (keys[0] if field_ == "host" else keys[1])
        else:
            rel[field_] = k
    if rel["host"] == rel["guest"]:
        rel["host"], rel["guest"] = keys[0], keys[1]
    rel.setdefault("params", {})
    guest_builder = next((c["builder"] for c in cs if c["key"] == rel["guest"]), None)
    if guest_builder == "solvent_box" and rel["kind"] in ("inside", "coated_by"):
        rel["kind"] = "around"            # a filled box can only SURROUND the host
        if (n := rel["params"].pop("count", None)):
            for c in cs:                  # the count means: n solvent molecules
                if c["key"] == rel["guest"]:
                    c["spec"].setdefault("n", n)
    host_c = next((c for c in cs if c["key"] == rel["host"]), None)
    guest_c = next((c for c in cs if c["key"] == rel["guest"]), None)
    # a sandwich implies SURFACES: a bulk host has no face to confine a film
    # against — convert it to a slab of the same material (default termination)
    if rel["kind"] == "between" and host_c and host_c["builder"] == "bulk":
        host_c["builder"] = "surface_slab"
        mat = host_c["spec"].get("material") or host_c["spec"].get("element")
        host_c["spec"] = {"element": mat}
    # "a dimer of two NPs" / "bcc superlattice of 2 NPs": the LLM invents a
    # between-relation among the particles — arrangement is the cluster's job
    if rel["kind"] == "between" and host_c \
            and host_c["builder"] == "nanoparticle":
        return None
    # a count the LLM parked on a MOLECULE constituent is invisible (not a
    # registry param there) — it belongs to the relation
    if guest_c and guest_c["builder"] == "molecule":
        for key_ in ("count", "n"):
            v = guest_c["spec"].pop(key_, None)
            if v and not rel["params"].get("count"):
                try:
                    rel["params"]["count"] = int(v)
                except (TypeError, ValueError):
                    pass
    if host_c and guest_c and host_c["builder"] == "surface_slab":
        # a slab is coated by FILLING ITS OWN CELL — not solvated in a separate
        # box; a liquid guest becomes plain molecules, and an LLM-invented n=1
        # is dropped (auto-fill at liquid density unless the user gave a count).
        # kept as-is: "between" (sandwich film) and "on" with an explicit SMALL
        # count (true adsorption geometry); "on" with 300 molecules is a film.
        n_rel = rel["params"].get("count")
        small_adsorption = rel["kind"] == "on" and n_rel and int(n_rel) <= 4
        if rel["kind"] != "between" and not small_adsorption:
            rel["kind"] = "coated_by"
        if guest_c["builder"] == "solvent_box":
            n = guest_c["spec"].get("n")
            guest_c["builder"] = "molecule"
            guest_c["spec"] = {"name": guest_c["spec"].get("molecule", "water")}
            if n and int(n) > 1:
                rel["params"]["count"] = int(n)
        if rel["kind"] == "coated_by":
            rel["params"].setdefault("count", None)       # None = fill the cell
    return rel


def _sanitize(state: dict) -> dict:
    """Registry is the single source of truth — drop anything outside it."""
    cs = [c for c in state.get("constituents") or []
          if c.get("builder") in BUILDERS]
    for c in cs:
        c["key"] = slug(c.get("key") or c["builder"])
        c.setdefault("spec", {})
        for k, v in list(c["spec"].items()):
            # an EXPLICIT null on a wulff gamma means "remove the facet", but
            # null also reads as "unset" (defaults would refill it) — remap to
            # the string marker the template understands
            if k.startswith("gamma_") and v is None:
                c["spec"][k] = "none"
            # cubic symmetry: (001)=(010)=(100) and (101)=(011)=(110) — an LLM
            # edit like "gamma_001: 2.0" must land on the canonical family
            # param, not invent a conflicting fourth facet
            m = re.fullmatch(r"gamma_(\d{3})", k)
            if m:
                canon = "gamma_" + "".join(sorted(m.group(1), reverse=True))
                if canon != k:
                    c["spec"][canon] = c["spec"].pop(k)
        # the LLM confuses the NP SHAPE (wulff/sphere/cube) with the
        # superlattice PACKING (sc/fcc/bcc) — remap to the right field
        if c["builder"] == "nanoparticle" \
                and str(c["spec"].get("shape") or "").lower() in ("sc", "fcc", "bcc"):
            c["spec"]["lattice"] = c["spec"].pop("shape").lower()
        # a slab repeat is in-plane NxM — the LLM cross-contaminates the bulk
        # builder's NxMxK (e.g. "2x2x2"), which would silently shrink the slab
        # and block hetero-sandwich auto-matching
        if c["builder"] == "surface_slab" and c["spec"].get("repeat"):
            if not re.fullmatch(r"\s*\d+\s*[x×]\s*\d+\s*",
                                str(c["spec"]["repeat"])):
                c["spec"].pop("repeat")
    keys = [c["key"] for c in cs]
    rels_in = state.get("relations")
    if rels_in is None:                   # legacy single-relation shape
        rels_in = [state["relation"]] if state.get("relation") else []
    rels, seen = [], set()
    for rel in rels_in:
        r = _sanitize_relation(rel, cs, keys)
        if r and (r["host"], r["guest"]) not in seen:
            seen.add((r["host"], r["guest"]))
            rels.append(r)

    # sandwich() builds the second slab ITSELF: an LLM that models "between
    # two slabs" as two slab constituents (each with its own between-relation
    # to the same guest) collapses to ONE relation. Identical slabs -> the
    # duplicate constituent is dropped (homo-sandwich); different slabs ->
    # the second becomes params.second_host (hetero-sandwich).
    by_key = {c["key"]: c for c in cs}
    kept, betw_by_guest, dropped = [], {}, set()
    for r in rels:
        if r["kind"] == "between":
            first = betw_by_guest.get(r["guest"])
            if first is not None:
                h1 = json.dumps((by_key.get(first["host"]) or {}).get("spec"),
                                sort_keys=True, default=str)
                h2 = json.dumps((by_key.get(r["host"]) or {}).get("spec"),
                                sort_keys=True, default=str)
                if h1 == h2:
                    dropped.add(r["host"])            # same slab twice
                else:
                    first["params"]["second_host"] = r["host"]
                # merge an explicit count (>1; a bare 1 is the usual
                # "put water molecules" hallucination)
                c2 = (r.get("params") or {}).get("count")
                if isinstance(c2, (int, float)) and c2 > 1 \
                        and not first["params"].get("count"):
                    first["params"]["count"] = int(c2)
                continue
            if (r.get("params") or {}).get("count") == 1:
                r["params"]["count"] = None           # n=1 hallucination
            betw_by_guest[r["guest"]] = r
        kept.append(r)
    rels = kept
    still_used = {r["host"] for r in rels} | {r["guest"] for r in rels} \
        | {r["params"]["second_host"] for r in rels
           if r.get("params", {}).get("second_host")}
    cs = [c for c in cs if c["key"] not in dropped or c["key"] in still_used]

    return {"constituents": cs, "relations": rels,
            "relation": rels[0] if rels else None,        # compat alias
            "summary": state.get("summary") or ""}


def parse_query(query: str, model: str = DEFAULT_MODEL) -> dict:
    """User query -> typed constituents + relation (LLM; keyword fallback keyless)."""
    if not have_openai_key():
        return _fallback_parse(query)
    state = chat_json([{"role": "user", "content": query}], _parse_system(), model=model)
    state = _sanitize(state)
    _apply_nparticles_hint(state, query)
    _strip_uninvited_repeat(state, query)
    _bulk_vs_slab_hint(state, query)
    return state


def _bulk_vs_slab_hint(state: dict, query: str) -> None:
    """'GaAs 110' means a SURFACE — a bare Miller index implies a termination.
    The LLM sometimes routes material+digits to the bulk builder; unless the
    user actually said bulk/crystal, convert it to a slab with that Miller."""
    if re.search(r"\b(bulk|crystal|supercell)\b", query, re.IGNORECASE):
        return
    m = re.search(r"\b(\d{3,4})\b", query)
    if not m:
        return
    for c in state.get("constituents", []):
        if c.get("builder") == "bulk":
            mat = c["spec"].get("material") or c["spec"].get("element")
            c["builder"] = "surface_slab"
            c["spec"] = {"element": mat, "miller": m.group(1)}


def apply_answer(state: dict, answer: str, gap: Gap | None = None,
                 model: str = DEFAULT_MODEL) -> dict:
    """Merge the user's message into the spec state (optionally answering `gap`)."""
    if not have_openai_key():
        return _fallback_apply(state, gap, answer)
    asked = (f"Open question (about '{gap.target}', field '{gap.param}'):\n"
             f"{gap.question}\n\n" if gap else "")
    user = (f"Current spec:\n{json.dumps(state)}\n\n{asked}"
            f"User's message: {answer}")
    return _sanitize(chat_json([{"role": "user", "content": user}],
                               _apply_system(), model=model))


_ROUTE_SHAPE = """Return ONLY JSON of this exact shape:
{"intent": "edit" | "question" | "build" | "new",
 "answer": "<for question: a short, direct, factual answer; else ''>",
 "state": <for edit: the full updated spec (same shape as the one given); else null>}"""


def _respond_system() -> str:
    return f"""You are the chat brain of a molecular-structure builder. The user has a
current build spec and the exact code that will run. Classify their message:

- "question": they ask ABOUT the plan/structure/parameters/methods (e.g. "is it a
  Wulff construction?", "what are the silica surface specifications?"). Answer like
  a precise materials scientist, FROM the spec, code, helper sources, and provenance
  given: state the concrete numbers and names found there — space group, lattice
  parameters (a, c, internal coordinates), Miller termination, slab thickness/width,
  atom counts and composition, literature sources cited in docstrings, surface-energy
  tables. NEVER answer vaguely or with "handled internally": if the provided material
  contains the values, quote them; if it truly doesn't, say exactly what is missing.
  Do NOT modify the spec.
- "edit": they request a change TO THE CURRENT system. Merge ONLY what their message
  determines into the spec. Counts: for a solvent_box guest, a NUMBER of solvent
  molecules (e.g. "5 water molecules") goes into THAT constituent's spec field "n";
  for a molecule guest of an inside/coated_by relation it goes to
  relation.params.count. "inside N water molecules" around a SOLID host (nanoparticle,
  slab) still means SOLVATED — keep relation kind "around"; never flip the relation so
  a liquid contains the solid unless the host is hollow (e.g. a nanotube). KEEP every
  constituent "key" exactly as given; keep relation.host/guest pointing at those keys.
  Rules: 1 nm = 10 Angstrom; diameter = 2 * radius. "add N more <guest>" for a guest
  that already has a relation means INCREASE that relation's params.count by N (an
  explicit user count, never a hallucination to drop). If they change parameters AND
  ask to build in the same message, intent is still "edit" — never build with a stale
  spec.
- "build": they ask to build / proceed / go ahead, WITHOUT requesting any change.
- "new": they describe a DIFFERENT system whose constituents don't match the current
  ones (e.g. current is a magnetite nanoparticle in water and they say "an aluminum
  surface coated with water") — do not merge; the request will be re-parsed fresh.

Builders and their spec fields:
{catalog()}

Relations between two constituents:
{relation_catalog()}
{_ROUTE_SHAPE}"""


def respond(state: dict, message: str, gap: Gap | None = None, context: str = "",
            model: str = DEFAULT_MODEL) -> dict:
    """Route a chat message: question -> answer; edit -> merged spec; build.

    Returns {"intent", "answer", "state"} — state is always a sanitized spec
    (unchanged unless intent == "edit").
    """
    if not have_openai_key():
        return {"intent": "edit", "answer": "",
                "state": _fallback_apply(state, gap, message)}
    asked = (f"Open point (about '{gap.target}', field '{gap.param}'):\n"
             f"{gap.question}\n\n" if gap else "")
    user = (f"Current spec:\n{json.dumps(state)}\n\n"
            + (f"Code that will run:\n{context}\n\n" if context else "")
            + f"{asked}User's message: {message}")
    out = chat_json([{"role": "user", "content": user}], _respond_system(), model=model)
    intent = out.get("intent") \
        if out.get("intent") in ("edit", "question", "build", "new") else "edit"
    if intent != "new" and _new_system_hint(state, message):
        intent = "new"
    new_state = state
    if intent == "edit":
        new_state = _sanitize(out.get("state") or {})
        if not new_state["constituents"]:
            new_state = state
        _apply_repeat_hint(new_state, message)
        _apply_facet_hint(new_state, message)
        _apply_nparticles_hint(new_state, message)
        _apply_add_count_hint(state, new_state, message)
        _strip_uninvited_repeat(new_state, message,
                                keep=_slab_repeats_of(state))
    return {"intent": intent, "answer": str(out.get("answer") or ""),
            "state": new_state}


def _apply_repeat_hint(state: dict, message: str) -> None:
    """Deterministic NxM supercell edit: the LLM reliably misreads '4x3' (it
    lands in width or nowhere), so a bare NxM in the user's message sets the
    slab's `repeat` directly. Skipped for NxMxK (a 3D box, not a supercell)
    and when a unit follows (e.g. '40x40 Å' is a size, not a repeat)."""
    m = re.search(r"\b(\d+)\s*[x×]\s*(\d+)\b"
                  r"(?!\s*[x×]\s*\d)(?!\s*(?:nm|Å|A\b|angstrom))",
                  message, re.IGNORECASE)
    if not m:
        return
    for c in state.get("constituents", []):
        if c.get("builder") == "surface_slab":
            c["spec"]["repeat"] = f"{m.group(1)}x{m.group(2)}"


# cubic families: any member index the user types maps to its gamma param
_FACET_FAMILY = {"111": "gamma_111",
                 "100": "gamma_100", "010": "gamma_100", "001": "gamma_100",
                 "110": "gamma_110", "101": "gamma_110", "011": "gamma_110"}


def _apply_facet_hint(state: dict, message: str) -> None:
    """Deterministic 'only these facets' edit: 'I want only 111 and 100 facets'
    must set every UNMENTIONED family's gamma to 'none' — the LLM reliably
    leaves the spec unchanged for this phrasing (removal by omission is
    invisible to a merge)."""
    if not re.search(r"\bonly\b", message, re.IGNORECASE) \
            or "facet" not in message.lower():
        return
    wanted = {_FACET_FAMILY[m] for m in re.findall(r"\b(\d{3})\b", message)
              if m in _FACET_FAMILY}
    if not wanted:
        return
    from .registry import BUILDERS
    defaults = {p.name: p.default for p in BUILDERS["nanoparticle"].params
                if p.name.startswith("gamma_")}
    for c in state.get("constituents", []):
        if c.get("builder") != "nanoparticle":
            continue
        for g, default in defaults.items():
            if g in wanted:                        # re-enable if previously off
                v = c["spec"].get(g)
                if v is None or not _is_number(v) or float(v) <= 0:
                    c["spec"][g] = default
            else:
                c["spec"][g] = "none"


def _is_number(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _slab_repeats_of(state: dict) -> dict:
    return {c["key"]: c["spec"].get("repeat")
            for c in state.get("constituents", [])
            if c.get("builder") == "surface_slab" and c["spec"].get("repeat")}


def _strip_uninvited_repeat(state: dict, text: str, keep: dict | None = None) -> None:
    """The LLM invents slab repeats ("2x2") the user never asked for — they
    silently shrink slabs and block hetero-sandwich lattice auto-matching.
    Drop any slab repeat that isn't in the user's text and wasn't already in
    the pre-edit state (`keep`)."""
    if re.search(r"\b\d+\s*[x×]\s*\d+\b", text):
        return                       # the user DID give an NxM — trust the spec
    keep = keep or {}
    for c in state.get("constituents", []):
        if c.get("builder") == "surface_slab" and c["spec"].get("repeat"):
            if c["spec"]["repeat"] != keep.get(c["key"]):
                c["spec"].pop("repeat")


_BUILDER_NOUNS = {"surface": "surface_slab", "slab": "surface_slab",
                  "nanoparticle": "nanoparticle", "np": "nanoparticle",
                  "nanotube": "nanotube", "cnt": "nanotube",
                  "bulk": "bulk", "crystal": "bulk"}


def _new_system_hint(state: dict, message: str) -> bool:
    """A message naming a builder TYPE the current system doesn't have
    ("magnetite surface and 50 oleic acid molecules on top" while building a
    nanoparticle) is a NEW system — the LLM tends to misroute it as an edit
    of the current one. Messages that mention a CURRENT builder noun or start
    with a modification verb ("add a gold surface on top") stay edits."""
    words = re.findall(r"[a-z]+", message.lower())
    if not words:
        return False
    current = {c.get("builder") for c in state.get("constituents", [])}
    named = {b for w, b in _BUILDER_NOUNS.items() if w in words}
    if not (named - current):
        return False                     # nothing new named
    if any(w in words for w, b in _BUILDER_NOUNS.items() if b in current):
        return False                     # they reference the current system
    if words[0] in {"add", "also", "put", "place", "coat", "cover", "make",
                    "set", "change", "increase", "decrease", "remove", "keep",
                    "use", "give", "fill", "solvate"}:
        return False                     # modification verbs anchor to it
    return True


_MOLAR = {"water": 18.02, "ethanol": 46.07, "methanol": 32.04, "hexane": 86.18,
          "toluene": 92.14, "benzene": 78.11, "acetone": 58.08, "ammonia": 17.03}
_NATOMS = {"water": 3, "ethanol": 9, "methanol": 6, "hexane": 20, "toluene": 15,
           "benzene": 12, "acetone": 10, "ammonia": 4}


def _apply_add_count_hint(old_state: dict, new_state: dict, message: str) -> None:
    """Deterministic increments: 'add N (more) water' or 'increase the water
    molecules by N' on an EXISTING guest — the LLM merge regularly no-ops
    these. Molecule guests bump the relation count; solvent_box guests get an
    explicit n (current effective fill + N) and a box grown to hold it."""
    m = re.search(r"\badd\s+(\d+)\s+(?:more\s+)?(\w+)", message, re.IGNORECASE)
    if m:
        n, name = int(m.group(1)), m.group(2).lower().rstrip("s")
    else:
        m = re.search(r"\bincrease\s+(?:the\s+)?(\w+)[\w\s]*?\bby\s+(\d+)",
                      message, re.IGNORECASE)
        if not m:
            return
        name, n = m.group(1).lower().rstrip("s"), int(m.group(2))
    old_counts = {(r["host"], r["guest"]): (r.get("params") or {}).get("count")
                  for r in old_state.get("relations") or []}
    for rel in new_state.get("relations") or []:
        gc = next((c for c in new_state.get("constituents", [])
                   if c["key"] == rel["guest"]), None)
        spec = (gc or {}).get("spec", {})
        gname = str(spec.get("name") or spec.get("molecule") or rel["guest"]).lower()
        if name not in gname and name not in rel["guest"].lower():
            continue
        if gc and gc.get("builder") == "solvent_box":
            mol = str(spec.get("molecule") or "water").lower()
            from .solvent import SOLVENT_DENSITY
            rho = SOLVENT_DENSITY.get(mol, 1.0)
            molar = _MOLAR.get(mol, 18.02)
            natoms = _NATOMS.get(mol, 3)
            box = float(spec.get("box_size") or 40.0)
            old_gc = next((c for c in old_state.get("constituents", [])
                           if c["key"] == rel["guest"]), None)
            old_n = (old_gc or {}).get("spec", {}).get("n")
            if old_n and spec.get("n") not in (old_n, None):
                return                             # the LLM already applied it
            if old_n:
                base = int(old_n)
            else:                                  # density autofill, capped
                dens_n = round(rho * (box * 1e-8) ** 3 * 6.022e23 / molar)
                base = min(dens_n, max(1, 8000 // natoms))
            total = base + n
            spec["n"] = total
            need = (total * molar / (rho * 6.022e23)) ** (1.0 / 3.0) * 1e8
            spec["box_size"] = max(box, round(need, 1))
            return
        old_c = old_counts.get((rel["host"], rel["guest"]))
        new_c = (rel.get("params") or {}).get("count")
        if isinstance(old_c, (int, float)) and old_c \
                and new_c == old_c:                    # LLM left it unchanged
            rel.setdefault("params", {})["count"] = int(old_c) + n
        return


def _apply_nparticles_hint(state: dict, message: str) -> None:
    """Deterministic multi-NP supercrystal handling — the LLM reliably drops
    the count and the packing, and invents 'supercrystal' as its own bulk
    constituent. 'a supercrystal of 10 nanoparticles', 'FCC supercrystal made
    of magnetite nanoparticles' (count asked later), 'cluster of 8 NPs'..."""
    nps = [c for c in state.get("constituents", [])
           if c.get("builder") == "nanoparticle"]
    if not nps:
        return
    m = re.search(r"\b(\d+)\s+(?!nm\b|angstrom|Å|A\b)"
                  r"(?:(?!nm\b|angstrom)\w+\s+){0,2}?"
                  r"(?:nanoparticles?|particles?|nps?)\b",
                  message, re.IGNORECASE)
    if not m:
        m = re.search(r"\b(?:cluster|supercrystal|superlattice|assembly)\s+"
                      r"(?:of|with|made of)\s+(\d+)\b", message, re.IGNORECASE)
    n = int(m.group(1)) if m else None
    lat = re.search(r"\b(fcc|bcc)\b", message, re.IGNORECASE)
    stk = re.search(r"\b([ABC]{4,})\b", message)     # explicit stacking sequence
    multi = re.search(r"\bsupercrystal|superlattice|nanoparticle cluster"
                      r"|cluster of nanoparticles\b", message, re.IGNORECASE)
    for c in nps:
        if n and n >= 2:
            c["spec"]["n_particles"] = n
        elif multi and int(c["spec"].get("n_particles") or 1) < 2:
            # stacking sequences default to 3 particles per layer
            c["spec"]["n_particles"] = 3 * len(stk.group(1)) if stk else 4
        if stk and (multi or (n and n >= 2)):
            c["spec"]["lattice"] = stk.group(1)
        elif lat and (multi or (n and n >= 2)):
            c["spec"]["lattice"] = lat.group(1).lower()
    if multi or (n and n >= 2):
        # "supercrystal"/"cluster" describes the NP arrangement — the LLM
        # sometimes also invents it as a SEPARATE bulk constituent
        bogus = {c["key"] for c in state.get("constituents", [])
                 if c.get("builder") == "bulk" and re.search(
                     r"super|cluster|lattice|assembl", c["key"], re.IGNORECASE)}
        if bogus:
            state["constituents"] = [c for c in state["constituents"]
                                     if c["key"] not in bogus]
            state["relations"] = [r for r in state.get("relations") or []
                                  if r["host"] not in bogus
                                  and r["guest"] not in bogus]
            state["relation"] = (state["relations"][0]
                                 if state["relations"] else None)


# ----------------------- keyless fallbacks (demo path) -------------------------

_MOLECULES = ("methanol", "ethanol", "water", "oleic acid", "hexane", "toluene",
              "benzene", "acetone", "ammonia")
_OXIDE_NPS = ("magnetite", "iron oxide", "titania", "silica")
_METAL_NPS = ("gold", "silver", "copper", "platinum", "palladium", "nickel")


def _size_A(text: str) -> float | None:
    m = re.search(r"(radius|diameter)\s*(?:of\s*)?(\d+(?:\.\d+)?)\s*(nm|a|angstrom|å)", text)
    if not m:
        m2 = re.search(r"(\d+(?:\.\d+)?)\s*(nm|a|angstrom|å)\s*(radius|diameter)?", text)
        if not m2:
            return None
        kind, val, unit = m2.group(3) or "diameter", float(m2.group(1)), m2.group(2)
    else:
        kind, val, unit = m.group(1), float(m.group(2)), m.group(3)
    v = val * 10.0 if unit == "nm" else val
    return 2 * v if kind == "radius" else v


def _fallback_parse(query: str) -> dict:
    t = query.lower()
    cs: list[dict] = []
    if "nanotube" in t or re.search(r"\bcnt\b", t):
        spec: dict = {}
        cm = re.search(r"\(?(\d+)\s*,\s*(\d+)\)?", t)
        if cm:
            spec["n"], spec["m"] = int(cm.group(1)), int(cm.group(2))
        cs.append({"key": "nanotube", "builder": "nanotube", "spec": spec})
    for name in (*_OXIDE_NPS, *_METAL_NPS):
        if name in t:
            spec = {"material": name}
            if (d := _size_A(t)) is not None:
                spec["diameter"] = d
            cs.append({"key": slug(name) + "_np", "builder": "nanoparticle", "spec": spec})
            break
    for name in _MOLECULES:
        if name in t:
            solvent = bool(re.search(rf"(in|solvat\w*|surround\w*)\s+{name}", t))
            if solvent and "inside" not in t:
                cs.append({"key": slug(name) + "_box", "builder": "solvent_box",
                           "spec": {"molecule": name}})
            else:
                cs.append({"key": slug(name), "builder": "molecule", "spec": {"name": name}})
            break
    rel = None
    if len(cs) >= 2:
        host = next((c for c in cs if c["builder"] in
                     ("nanotube", "nanoparticle", "surface_slab")), cs[0])
        guest = next(c for c in cs if c is not host)
        kind = None
        if "inside" in t:
            kind = "inside"
        elif guest["builder"] == "solvent_box" or "solvat" in t or "surround" in t:
            kind = "around"
        elif "coat" in t or "graft" in t:
            kind = "coated_by"
        elif re.search(r"\bon\b|adsorb", t):
            kind = "on"
        if kind:
            rel = {"kind": kind, "host": host["key"], "guest": guest["key"], "params": {}}
    return _sanitize({"constituents": cs, "relation": rel,
                      "summary": f"(keyword parse) {len(cs)} constituent(s)"})


def _fallback_apply(state: dict, gap: Gap | None, answer: str) -> dict:
    st = json.loads(json.dumps(state))
    # deep copy severs the relation/relations[0] alias — re-link so a mutation
    # through either name is seen by both readers
    rels = st.get("relations") or ([st["relation"]] if st.get("relation") else [])
    st["relations"], st["relation"] = rels, (rels[0] if rels else None)

    def _set_rel(rel: dict) -> None:
        st["relations"], st["relation"] = [rel], rel

    a = answer.strip().lower()
    use_default = "default" in a or a in ("yes", "y", "ok", "sure", "")

    if gap is None:                    # keyless free-form: only relation keywords resolve
        kind = next((k for k in RELATIONS if k.replace("_", " ") in a or k in a), None)
        if kind and len(st["constituents"]) >= 2 and not st.get("relation"):
            ks = [c["key"] for c in st["constituents"]]
            _set_rel({"kind": kind, "host": ks[0], "guest": ks[1], "params": {}})
        return st

    if gap.target == "relation":
        rel = st.get("relation") or {}
        if gap.param == "kind":
            kind = next((k for k in RELATIONS if k.replace("_", " ") in a or k in a), None)
            if kind and len(st["constituents"]) >= 2:
                ks = [c["key"] for c in st["constituents"]]
                _set_rel({"kind": kind, "host": ks[0], "guest": ks[1], "params": {}})
        elif (m := re.search(r"\d+", a)):
            rel.setdefault("params", {})["count"] = int(m.group())
        elif use_default:
            rel.setdefault("params", {})["count"] = None
        return st

    c = next((c for c in st["constituents"] if c["key"] == gap.target), None)
    if c is None:
        return st
    p = next((p for p in BUILDERS[c["builder"]].params if p.name == gap.param), None)
    if p is None:
        return st
    if gap.param == "n" and (cm := re.search(r"(\d+)\s*,\s*(\d+)", a)):
        c["spec"]["n"], c["spec"]["m"] = int(cm.group(1)), int(cm.group(2))
    elif use_default and p.default is not None:
        c["spec"][gap.param] = p.default
    elif p.kind == "bool":
        c["spec"][gap.param] = a.startswith(("y", "true", "1"))
    elif p.kind in ("float", "int") and (m := re.search(r"-?\d+(?:\.\d+)?", a)):
        c["spec"][gap.param] = float(m.group()) if p.kind == "float" else int(float(m.group()))
    elif p.kind == "str" and a and not use_default:
        c["spec"][gap.param] = answer.strip()
    elif use_default:
        c["spec"][gap.param] = p.default
    return st
