"""Moltemplate Knowledge Graph — engine (graph, retrieval, Gate 1 validator).

GENIUS-style grounding for Moltemplate:
  - retrieve(request): keyword-triggered conditions + graph expansion -> which
    constructs are relevant / allowed / excluded for a request (feeds generation).
  - validate_lt(text): STATIC anti-hallucination gate (Gate 1) -> every command,
    section, transform (with arg count) and variable in generated .lt must exist in
    the KG. Unknown constructs are flagged with a nearest-valid suggestion, BEFORE
    moltemplate.sh ever runs.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import get_close_matches

import networkx as nx

from . import kg_data as D


# --- graph construction -------------------------------------------------------

def build_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    for name, desc in D.COMMANDS.items():
        g.add_node(name, category="command", description=desc)
    for name, meta in D.TRANSFORMS.items():
        g.add_node(name, category="transform", description=meta["desc"], arities=meta["arities"])
    for name in D.FORMAT_METHODS:
        g.add_node(name, category="format_method", description="variable text formatting")
    for name, meta in D.DATA_SECTIONS.items():
        g.add_node(name, category="section", **meta)
    for name, desc in D.VARIABLES.items():
        g.add_node(name, category="variable", description=desc)
    for name, desc in D.PATH_TOKENS.items():
        g.add_node(name, category="path", description=desc)
    for src, dst, rel in D.EDGES:
        g.add_edge(src, dst, relation=rel)
    return g


# --- retrieval ----------------------------------------------------------------

@dataclass
class Retrieval:
    conditions: list[str]
    allowed: set[str]
    excluded: set[str]
    notes: list[str]


# --- Gate 1 validation --------------------------------------------------------

@dataclass
class Issue:
    severity: str          # "error" | "warning"
    construct: str
    message: str
    suggestion: str = ""


@dataclass
class ValidationReport:
    passed: bool
    issues: list[Issue] = field(default_factory=list)

    def errors(self) -> list[Issue]:
        return [i for i in self.issues if i.severity == "error"]

    def summary(self) -> str:
        if self.passed and not self.issues:
            return "Gate 1: PASS (all constructs grounded in KG)"
        head = "Gate 1: " + ("PASS with warnings" if self.passed else "FAIL")
        lines = [f"  [{i.severity}] {i.construct}: {i.message}"
                 + (f"  -> did you mean '{i.suggestion}'?" if i.suggestion else "")
                 for i in self.issues]
        return "\n".join([head, *lines])


_RE_TRANSFORM = re.compile(r"\.([A-Za-z_]\w*)\s*\(([^)]*)\)")
_RE_SECTION = re.compile(r'write(?:_once)?\s*\(\s*"([^"]+)"\s*\)')
_RE_VARIABLE = re.compile(r"([$@]\w+):")
_RE_NEWARRAY = re.compile(r"\bnew\b")


def _count_args(arg_str: str) -> int:
    parts = [a for a in arg_str.split(",") if a.strip()]
    return len(parts)


class MoltemplateKG:
    """The queryable Moltemplate knowledge graph."""

    def __init__(self) -> None:
        self.g = build_graph()
        self.transforms = set(D.TRANSFORMS)
        self.sections = set(D.DATA_SECTIONS)
        self.commands = set(D.COMMANDS)
        self.variables = set(D.VARIABLES)

    # -- stats
    def stats(self) -> dict:
        cats: dict[str, int] = {}
        for _, data in self.g.nodes(data=True):
            cats[data["category"]] = cats.get(data["category"], 0) + 1
        return {"nodes": self.g.number_of_nodes(), "edges": self.g.number_of_edges(),
                "conditions": len(D.CONDITIONS), "by_category": cats}

    # -- retrieval: request text -> relevant constructs
    def retrieve(self, request: str, geometry_only: bool = True) -> Retrieval:
        text = request.lower()
        active, allowed, excluded, notes = [], set(), set(), []
        if geometry_only:
            active.append("geometry_only")
        for name, cond in D.CONDITIONS.items():
            if name in active or any(t in text for t in cond["triggers"]):
                if name not in active:
                    active.append(name)
        for name in active:
            cond = D.CONDITIONS[name]
            allowed.update(cond["include"])
            excluded.update(cond["exclude"])
            notes.append(f"[{name}] {cond['note']}")
        # graph expansion: pull in 'requires' prerequisites of allowed sections
        for node in list(allowed):
            if self.g.has_node(node):
                for _, dst, d in self.g.out_edges(node, data=True):
                    if d.get("relation") == "requires":
                        allowed.add(dst)
        allowed -= excluded
        return Retrieval(conditions=active, allowed=allowed, excluded=excluded, notes=notes)

    # -- Gate 1: validate generated .lt against the KG
    def validate_lt(self, text: str, geometry_only: bool = True) -> ValidationReport:
        issues: list[Issue] = []

        for m in _RE_TRANSFORM.finditer(text):
            op, args = m.group(1), m.group(2)
            if op in D.FORMAT_METHODS or op in D.PLACEHOLDERS:
                continue                                   # valid non-transform / doc metavar
            if op not in self.transforms:
                sugg = get_close_matches(op, self.transforms | D.FORMAT_METHODS, n=1)
                issues.append(Issue("error", f".{op}", "unknown transform",
                                    sugg[0] if sugg else ""))
            else:
                arities = D.TRANSFORMS[op]["arities"]
                n = _count_args(args)
                if n not in arities:
                    want = "/".join(str(a) for a in sorted(arities))
                    issues.append(Issue("error", f".{op}",
                                        f"expects {want} args, got {n}"))

        for m in _RE_SECTION.finditer(text):
            sec = m.group(1)
            if sec not in self.sections:
                # near-match to a known section => likely a typo (error); otherwise it
                # may be a legitimate custom "Data ..." section (manual §5.6) => warning.
                close = get_close_matches(sec, self.sections, n=1, cutoff=0.8)
                if close:
                    issues.append(Issue("error", f'"{sec}"',
                                        "unknown data section (likely typo)", close[0]))
                else:
                    issues.append(Issue("warning", f'"{sec}"',
                                        "custom/unknown section (allowed if intentional; manual §5.6)"))
            elif geometry_only and not D.DATA_SECTIONS[sec]["geometry_ok"]:
                issues.append(Issue("warning", f'"{sec}"',
                                    "force-field section in geometry-only mode"))

        for m in _RE_VARIABLE.finditer(text):
            var = m.group(1)
            if var not in self.variables:
                sugg = get_close_matches(var, self.variables, n=1)
                issues.append(Issue("warning", var, "unknown variable sigil",
                                    sugg[0] if sugg else ""))

        passed = not any(i.severity == "error" for i in issues)
        return ValidationReport(passed=passed, issues=issues)
