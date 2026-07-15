"""Moltemplate Agent — KG-grounded structure builder (Streamlit chat).

Backbone (see .claude/skills/backbone/SKILL.md): the LLM only PARSES the query and
your replies; the KGs supply the real syntax. The first answer already shows the
suggested ASE/Moltemplate snippet for EVERY constituent (Gate 1 validated — that
exact code runs) with an explanation of each parameter, then asks how to continue:
adjust anything in plain language, answer the open points, or say "build".
Every constituent is built and shown; the combined cell is a SHOWCASE — a real
simulation cell needs MD equilibration, which is out of scope here.

Run:  conda activate mdagent && streamlit run app.py
"""
from __future__ import annotations

import importlib
import inspect
import json
import re
import traceback
from io import StringIO

import streamlit as st
import streamlit.components.v1 as components
from ase.io import write as ase_write

from mtagent import clarify, ground
from mtagent.assemble import RELATIONS
from mtagent.execute import run_snippet
from mtagent.propose import propose
from mtagent.registry import BUILDERS, slug
from mtagent.verify import verify_atoms
from mtagent.viewer import build_html

st.set_page_config(page_title="Atomistic Structure Builder", layout="wide")
st.markdown("""<style>
/* ChatGPT-like chat: no role avatars, user turns in a soft bubble */
[data-testid^="stChatMessageAvatar"], [data-testid^="chatAvatarIcon"] {
    display: none; }
.stChatMessage { gap: 0.25rem; }
.stChatMessage:has([data-testid="stChatMessageAvatarUser"]),
.stChatMessage:has([data-testid="chatAvatarIcon-user"]) {
    background: rgba(159, 180, 208, 0.12);   /* slate tint, scheme 9 */
    border-radius: 14px; padding: 10px 14px; }
/* don't dim/fade content while the script reruns */
[data-stale="true"] { opacity: 1 !important; pointer-events: auto !important; }
[data-testid="stStatusWidget"] { visibility: hidden; }
/* text must always be selectable/copyable — chat, tables, captions, everything */
.stChatMessage, .stChatMessage *, .stMarkdown, .stMarkdown *,
[data-testid="stExpander"] * {
    -webkit-user-select: text !important; user-select: text !important; }
</style>""", unsafe_allow_html=True)
APP_NAME = "Atomistic Structure Builder"
APP_VERSION = "0.1.0"

st.title(APP_NAME)

SS = st.session_state
for k, v in (("messages", []), ("spec", None), ("gap", None), ("final", None),
             ("proposals", None), ("results", None), ("prop_cache", {})):
    SS.setdefault(k, v)

_BUILD_WORDS = ("build", "go ahead", "proceed", "looks good", "do it")
_RESET_WORDS = ("start over", "new structure", "different structure", "forget", "reset")

_CHECK_NAMES = {"has_atoms": "atoms present", "finite_coords": "coordinates finite",
                "no_clash": "no atomic overlaps"}


def wants_build(text: str) -> bool:
    """Fast path only for short, pure build triggers ("build", "go ahead") —
    longer messages go through the LLM router so 'build it 6 nm instead'
    is applied as an edit first, never built with the stale spec."""
    t = text.strip().lower()
    return (any(w in t for w in _BUILD_WORDS) and "don't" not in t
            and "not " not in t and len(t.split()) <= 4)


def wants_reset(text: str) -> bool:
    return any(w in text.strip().lower() for w in _RESET_WORDS)


def checks_line(report) -> str:
    return " · ".join(("✓ " if c.passed else "✗ ")
                      + _CHECK_NAMES.get(c.name, c.name.replace("_", " "))
                      for c in report.checks)


def atoms_to_xyz(atoms) -> str:
    """Extended-xyz text: keeps the cell (Lattice=...) in the comment line.

    Multi-line info strings (e.g. packmol_inp) must be dropped first — extxyz
    writes them raw into the single-line comment, corrupting the file."""
    a = atoms.copy()
    a.info = {k: v for k, v in a.info.items()
              if not (isinstance(v, str) and "\n" in v)}
    s = StringIO()
    ase_write(s, a, format="extxyz")
    return s.getvalue()


def enclosure_split(atoms) -> tuple[int | None, str]:
    """Viewer split for ENCLOSURE systems: (atom count of the first component,
    which side renders translucent).

    Solvation shells and NP ligand shells surround the solute -> "tail";
    a filled tube's wall surrounds its guests -> "head". Interfaces
    (slab coats/adsorbates, mode "fill_cell") stay fully solid -> (None, _)."""
    sv = atoms.info.get("solvation")
    if sv and sv.get("solute_atoms"):
        return int(sv["solute_atoms"]), "tail"
    asm = atoms.info.get("assembly") or {}
    if asm.get("host_atoms"):
        if asm.get("relation") == "inside":
            return int(asm["host_atoms"]), "head"
        if asm.get("relation") == "coated_by" and asm.get("mode") == "shell":
            return int(asm["host_atoms"]), "tail"
    return None, "tail"


def gizmo_axes(atoms) -> list | None:
    """Axis-gizmo arrows. Slabs: the surface normal (+z) carries the Miller
    index of the termination, and the two in-plane arrows point along the
    actual surface-cell vectors labeled with their crystallographic [uvw]
    directions (recorded in provenance at build time). Others: cubic axes."""
    prov = atoms.info.get("provenance") or {}
    if prov.get("type") != "surface_slab" or not prov.get("miller"):
        return None

    def fmt(v) -> str:
        return "[" + "".join(str(int(x)) for x in v) + "]"

    normal = [fmt(prov["miller"]), [0, 0, 1]]
    uvw = prov.get("in_plane_uvw")
    if uvw and atoms.cell.volume > 1e-6:
        import numpy as np
        a, b = (np.asarray(atoms.cell[i], dtype=float) for i in (0, 1))
        return [[fmt(uvw[0]), list((a / np.linalg.norm(a)).round(4))],
                [fmt(uvw[1]), list((b / np.linalg.norm(b)).round(4))],
                normal]
    return [["", [1, 0, 0]], ["", [0, 1, 0]], normal]


# ------------------------- PROPOSE (cached per spec) ---------------------------

def proposal_for(target: str, evidence, task: str, cache_key: str):
    if cache_key not in SS.prop_cache:
        SS.prop_cache[cache_key] = propose(evidence, task)
    return SS.prop_cache[cache_key]


def make_proposals(final: dict) -> dict:
    props = {}
    for c in final["constituents"]:
        ck = json.dumps({"b": c["builder"], "s": c["spec"]}, sort_keys=True)
        props[c["key"]] = proposal_for(
            c["key"], ground.evidence_for(c),
            f"Build '{c['key']}': {c['builder']} with spec {json.dumps(c['spec'])}", ck)
    rels = ground.relations_of(final)
    if rels:
        ck = json.dumps({"rels": rels}, sort_keys=True)
        desc = "; then ".join(
            f"{r['guest']} {r['kind'].replace('_', ' ')} {r['host']} "
            f"(params {json.dumps(r.get('params') or {})})" for r in rels)
        props["showcase"] = proposal_for(
            "showcase", ground.evidence_for_relation(final),
            f"Combine, in order: {desc}. All constituent variables are "
            "already-built ase.Atoms.", ck)
    return props


# --------------------- the assistant's suggestion message ----------------------

def param_table(builder_name: str, raw_spec: dict, final_spec: dict) -> str:
    rows = ["| parameter | value | meaning |", "|---|---|---|"]
    for p in BUILDERS[builder_name].params:
        if p.when is not None and not p.when(final_spec):
            continue
        val = final_spec.get(p.name)
        if val is None and p.default is None and not p.required:
            continue
        shown = "**?**" if val is None else f"`{val}`"
        if val is not None and raw_spec.get(p.name) is None:
            shown += " *(default)*"
        rows.append(f"| `{p.name}` | {shown} | {p.help or p.ask} |")
    return "\n".join(rows)


def suggestion_message(state: dict, final: dict, proposals: dict) -> str:
    cs = final["constituents"]
    raw = {c["key"]: (c.get("spec") or {}) for c in state["constituents"]}
    names = " and ".join(f"**{c['key']}**" for c in cs)
    n = len(cs)
    out = [f"**{n} constituent{'s' if n > 1 else ''}**: {names}, "
           "here is the suggested build:"]

    for c in cs:
        p = proposals[c["key"]]
        out += [f"\n#### {c['key']}  ·  `{c['builder']}`",
                f"```python\n{p.code}\n```",
                param_table(c["builder"], raw.get(c["key"], {}), c["spec"])]
        if not p.report.passed:
            out += [f"Note, this snippet has validation issues: {p.report.summary()}"]

    rels = ground.relations_of(final)
    if rels and "showcase" in proposals:
        p = proposals["showcase"]
        out += ["\n#### combined structure: " + rel_label(final),
                f"```python\n{p.code}\n```",
                "*The combined cell is geometry only, a simulation ready cell "
                "needs MD equilibration, which is outside this agent's scope.*"]
    elif n >= 2:
        out += ["\n**How should these be combined?** " +
                ", ".join(f"`{k}`" for k in RELATIONS) +
                f" (e.g. *\"{cs[1]['key']} inside {cs[0]['key']}\"*)."]

    return "\n".join(out)


def spec_problem(state: dict) -> str | None:
    """Dry-run the canonical templates on the finalized spec. Parameter
    combinations the registry refuses (for example fewer than 2 Wulff facet
    families) surface HERE, as a chat reply, not as a build time traceback."""
    import copy
    try:
        final = ground.finalize(copy.deepcopy(state))
        for c in final["constituents"]:
            BUILDERS[c["builder"]].template(c["spec"])
    except ValueError as e:
        return str(e)
    return None


def refresh(state: dict) -> None:
    """Re-finalize + re-propose after every spec change, and post the suggestion."""
    SS.spec = state
    SS.final = ground.finalize(state)
    SS.proposals = make_proposals(SS.final)
    SS.results = None                  # spec changed -> previous build is stale
    gs = ground.gaps(state)
    SS.gap = gs[0].__dict__ if gs else None
    SS.messages.append({"role": "assistant",
                        "content": suggestion_message(state, SS.final, SS.proposals)})


# --------------------------- full build script ---------------------------------

def full_script(final: dict, proposals: dict) -> str:
    """One self-contained script reproducing everything that was built."""
    parts = ["# Python build script — Moltemplate Agent",
             "# Run from the project root (conda env mdagent):  python build_structure.py",
             "from ase.io import write", "",
             "def save(path, atoms):",
             "    # extxyz comments are single-line: multi-line info strings",
             "    # (e.g. the packmol input) would corrupt the .xyz file",
             "    a = atoms.copy()",
             "    a.info = {k: v for k, v in a.info.items()",
             "              if not (isinstance(v, str) and '\\n' in v)}",
             "    write(path, a)", ""]
    for c in final["constituents"]:
        k = c["key"]
        parts += [f"# ---- {k} ({c['builder']}) ----",
                  proposals[k].code,
                  f"{k} = atoms",
                  f'save("{k}.xyz", {k})', ""]
    if ground.relations_of(final) and "showcase" in proposals:
        parts += [f"# ---- combined structure: {rel_label(final)} ----",
                  proposals["showcase"].code,
                  'save("combined.xyz", atoms)', ""]
    return "\n".join(parts)


def helper_sources(proposals: dict) -> dict[str, str]:
    """Source of every mtagent function the snippets call, with the module-level
    constants it references (e.g. the Wulff surface-energy table) prepended."""
    out: dict[str, str] = {}
    for p in proposals.values():
        for mod, names in re.findall(r"from (mtagent\.\w+) import ([\w, ]+)", p.code):
            for name in (n.strip() for n in names.split(",")):
                try:
                    m = importlib.import_module(mod)
                    src = inspect.getsource(getattr(m, name))
                    consts = {c: getattr(m, c)
                              for c in sorted(set(re.findall(r"\b[A-Z][A-Z0-9_]{2,}\b", src)))
                              if hasattr(m, c) and not inspect.ismodule(getattr(m, c))}
                    if consts:
                        src = "\n".join(f"{k} = {v!r}" for k, v in consts.items()) \
                              + "\n\n" + src
                    out[f"{mod}.{name}"] = src
                except Exception:
                    pass
    return out


def qa_context() -> str:
    """Everything the Q&A may quote from — hard facts about the BUILT structures
    first (they must survive truncation), then snippets, then helper sources."""
    parts = []
    if SS.results:
        for k, (a, _) in SS.results.items():
            syms = a.get_chemical_symbols()
            comp = {e: syms.count(e) for e in sorted(set(syms))}
            facts = {"chemical_formula": a.get_chemical_formula(),
                     "composition": comp, "n_atoms": len(a)}
            if a.info.get("provenance"):
                facts["provenance"] = a.info["provenance"]
            parts.append(f"# built structure '{k}' — measured facts\n"
                         f"{json.dumps(facts, default=str)}")
    parts += [f"# {k}\n{p.code}" for k, p in (SS.proposals or {}).items()]
    parts += [f"# source of {qual}\n{src}"
              for qual, src in helper_sources(SS.proposals or {}).items()]
    return "\n\n".join(parts)[:24000]


def build_all() -> None:
    results, rels = {}, ground.relations_of(SS.final)
    with st.status("Building the structures…", expanded=True) as sb:
        for c in SS.final["constituents"]:
            key = c["key"]
            sb.write(f"**{key}**: building…")
            atoms = run_snippet(SS.proposals[key].code)               # Gate 2
            results[key] = (atoms, verify_atoms(atoms))               # Gate 3
            sb.write(f"**{key}**: {len(atoms)} atoms")
        if rels and "showcase" in SS.proposals:
            sb.write(f"Combining: {rel_label(SS.final)}…")
            inputs = {k: a for k, (a, _) in results.items()}
            atoms = run_snippet(SS.proposals["showcase"].code, inputs)
            results["showcase"] = (atoms, verify_atoms(atoms))
            sb.write(f"Combined: {len(atoms)} atoms")
        sb.update(label="Build finished", state="complete", expanded=False)
    SS.results = results
    built = ", ".join(
        f"**{rel_label(SS.final) if k == 'showcase' else display_name(k, a)}**"
        for k, (a, _) in results.items())
    SS.messages.append({"role": "assistant",
                        "content": f"Built {built}:",
                        "results": results,
                        "proposals": dict(SS.proposals),
                        "final": SS.final})


# ----------------------------------- UI ---------------------------------------

with st.sidebar:
    st.markdown(f"### {APP_NAME}")
    st.markdown("A knowledge graph grounded atomistic structure builder.")
    st.link_button("Documentation and examples",
                   "https://emregrsy.github.io/atomistic-structure-builder/",
                   use_container_width=True)

if not SS.messages:
    st.markdown(
        "Builds atomistic systems from plain language. Requests are grounded "
        "in knowledge graphs of ASE, packmol, Moltemplate and PubChem, so "
        "every build runs real, validated code, and the code you see is the "
        "code that runs.")
    st.markdown(
        '<div style="background:#222834; border:1px solid #39435a; '
        'border-radius:10px; padding:14px 18px; font-size:15px;">'
        'Tell me what to build, e.g. <i>"a carbon nanotube with methanol '
        'inside"</i> or <i>"a 4 nm magnetite nanoparticle in water"</i>. '
        'You immediately get the suggested build code for every constituent, '
        'with the parameters explained.</div>', unsafe_allow_html=True)

def display_name(key: str, atoms) -> str:
    """Slabs carry orientation + supercell, e.g. magnetite_slab (001) 3×3."""
    prov = atoms.info.get("provenance") or {}
    if prov.get("type") == "surface_slab" and prov.get("miller"):
        name = f"{key} ({''.join(str(i) for i in prov['miller'])})"
        if prov.get("repeat"):
            name += f" {prov['repeat'][0]}×{prov['repeat'][1]}"
        return name
    if prov.get("type") == "np_cluster":
        lat = prov.get("lattice", "sc")
        return f"{_nice_key(key)} supercrystal ({lat}, {prov.get('n_units')} NPs)"
    return key


def _nice_key(key: str) -> str:
    for suffix in ("_slab", "_surface", "_molecule", "_nanoparticle", "_np"):
        key = key.removesuffix(suffix)
    return key.replace("_", " ")


def rel_label(final: dict) -> str:
    """The combined structure, named by what was asked (never 'showcase').

    Slab + liquid reads as an INTERFACE ("magnetite ethanol interface"),
    a confined film as a sandwich; other relations keep guest-kind-host."""
    rels = ground.relations_of(final)
    if not rels:
        return "combined structure"
    by_key = {c["key"]: c for c in final["constituents"]}

    def one(r):
        host, guest = _nice_key(r["host"]), _nice_key(r["guest"])
        host_c = by_key.get(r["host"])
        slab_host = host_c and host_c.get("builder") == "surface_slab"
        if r["kind"] == "coated_by" and slab_host:
            return f"{host} {guest} interface"
        if r["kind"] == "between":
            top = (r.get("params") or {}).get("second_host")
            return (f"{host} {guest} {_nice_key(top)} sandwich" if top
                    else f"{host} {guest} sandwich")
        return f"{guest} {r['kind'].replace('_', ' ')} {host}"

    return " + ".join(one(r) for r in rels)


def render_results(m: dict, idx: int) -> None:
    """Structures rendered inline in the chat flow, anchored to their message."""
    results, proposals, final = m["results"], m["proposals"], m["final"]
    for key, (atoms, report) in results.items():
        is_showcase = key == "showcase"
        name = rel_label(final) if is_showcase else display_name(key, atoms)
        title = f"{name}, {len(atoms)} atoms" + \
            (" (geometry only, NOT equilibrated)" if is_showcase else "")
        with st.expander(title, expanded=is_showcase):
            st.code(proposals[key].code, language="python")
            if len(atoms) == 0:
                st.warning("This build produced 0 atoms. The parameters carve "
                           "or filter everything away. Adjust them and rebuild.")
                continue
            if atoms.info.get("packmol_inp"):
                st.markdown("**packmol packing script** (the exact input used):")
                st.code(atoms.info["packmol_inp"], language="text")
            xyz = atoms_to_xyz(atoms)
            if atoms.cell.volume > 1e-6:            # draw the box AROUND the atoms
                cell = atoms.cell[:]
                origin = (atoms.get_positions().mean(axis=0)
                          - cell.sum(axis=0) / 2.0)
            else:                                   # no cell (NP/cluster): bounding
                import numpy as np                  # box + 5 A vacuum margin
                p = atoms.get_positions()
                ext = p.max(axis=0) - p.min(axis=0) + 10.0
                cell = np.diag(ext)
                origin = p.min(axis=0) - 5.0
            split, trans_side = enclosure_split(atoms)
            components.html(build_html(xyz, title=name, height=430, cell=cell,
                                       cell_origin=origin, split=split,
                                       translucent=trans_side,
                                       axes=gizmo_axes(atoms)), height=505)
            fname = slug(name) + ".xyz"
            st.download_button(f"Download {fname}", xyz, file_name=fname,
                               mime="text/plain", key=f"dl_{idx}_{key}")
            if atoms.info.get("lt_files"):
                for fname, text in atoms.info["lt_files"].items():
                    st.download_button(f"Download {fname}", text, file_name=fname,
                                       mime="text/plain", key=f"dl_{idx}_{key}_{fname}")
    script = full_script(final, proposals)
    with st.expander("Python build script"):
        st.code(script, language="python")
        st.download_button("Download build_structure.py", script,
                           file_name="build_structure.py", mime="text/x-python",
                           key=f"dl_script_{idx}")
        st.markdown("**Source of every helper the script calls:**")
        for qualname, src in helper_sources(proposals).items():
            with st.popover(qualname):
                st.code(src, language="python")


for i, m in enumerate(SS.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("error"):
            st.code(m["error"])        # st.code has a copy button
        if m.get("results"):
            render_results(m, i)


def parse_fresh(prompt: str, sb) -> None:
    sb.write("Parsing the request…")
    state = clarify.parse_query(prompt)
    if not state["constituents"]:
        SS.messages.append({"role": "assistant", "content":
                            "I couldn't map that to any known builder "
                            f"({', '.join(BUILDERS)}). Try naming the structures "
                            "directly, e.g. *\"a carbon nanotube with methanol\"*."})
        return
    names = ", ".join(c["key"] for c in state["constituents"])
    sb.write(f"Constituents: **{names}**")
    sb.write("Retrieving knowledge graph evidence, writing and validating "
             "the build code…")
    refresh(state)
    sb.write("Done.")


prompt = st.chat_input("Describe the structure, adjust parameters, or say 'build'…")
if not prompt:
    prompt = SS.pop("queued_prompt", None)       # sidebar example click
if prompt:
    SS.messages.append({"role": "user", "content": prompt})
    try:
        if SS.spec is None or not SS.spec["constituents"] or wants_reset(prompt):
            for k in ("spec", "gap", "final", "proposals", "results"):
                SS[k] = None
            with st.status("Working on it…", expanded=True) as sb:
                parse_fresh(prompt, sb)
                sb.update(label="Done", state="complete", expanded=False)
        elif wants_build(prompt):
            build_all()
        else:
            with st.status("Working on it…", expanded=True) as sb:
                sb.write("Routing your message…")
                gap = ground.Gap(**SS.gap) if SS.gap else None
                r = clarify.respond(SS.spec, prompt, gap, qa_context())
                if r["intent"] == "question":
                    sb.write("Answering…")
                    SS.messages.append({"role": "assistant",
                                        "content": r["answer"] or "I'm not sure, "
                                        "could you rephrase that?"})
                elif r["intent"] == "new":
                    sb.write("New system, parsing fresh…")
                    for k in ("spec", "gap", "final", "proposals", "results"):
                        SS[k] = None
                    parse_fresh(prompt, sb)
                elif r["intent"] == "edit" and json.dumps(
                        r["state"], sort_keys=True) != json.dumps(SS.spec, sort_keys=True):
                    problem = spec_problem(r["state"])
                    if problem:
                        SS.messages.append({
                            "role": "assistant",
                            "content": f"I can't apply that: {problem} "
                                       "The spec stays as shown above."})
                    else:
                        sb.write("Applying changes…")
                        refresh(r["state"])
                elif r["intent"] == "edit":
                    SS.messages.append({"role": "assistant", "content":
                                        r.get("note") or
                                        "That didn't change anything in the plan. The "
                                        "spec is as shown above. Ask me a question about "
                                        "it, adjust a parameter, or say **build**."})
                sb.update(label="Done", state="complete", expanded=False)
            if r["intent"] == "build":
                build_all()
    except Exception as e:
        SS.messages.append({"role": "assistant",
                            "content": f"That failed: **{type(e).__name__}** — full "
                                       "error below (copy button in the top-right of the "
                                       "box). You can adjust and try again.",
                            "error": traceback.format_exc()})
    st.rerun()

# ------------------------------ build button ----------------------------------
if SS.final is not None and SS.results is None:
    if st.button("Build", type="primary"):
        try:
            build_all()
        except Exception:
            SS.messages.append({"role": "assistant",
                                "content": "The build failed — full error below.",
                                "error": traceback.format_exc()})
        st.rerun()
