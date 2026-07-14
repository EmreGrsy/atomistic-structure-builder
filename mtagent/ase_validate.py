"""Gate 1 for the ASE agent — AST validation of generated Python against the ASE KG.

Parses generated code, resolves ASE calls via their imports, and checks each against the
real introspected signature:
  - the function/class exists in ASE          (catches hallucinated names)
  - keyword arguments are real parameters      (catches hallucinated kwargs)
  - positional-argument count is within bounds (catches arity mistakes)

This is stronger than the Moltemplate regex validator because Python is introspectable.
Method calls on variables (e.g. atoms.foo()) are not resolved statically -> skipped to
avoid false positives; the high-value function/constructor calls are checked strictly.
"""
from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass, field
from difflib import get_close_matches
from typing import Iterable

from .ase_kg import ASEKnowledgeGraph


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

    def summary(self) -> str:
        if self.passed and not self.issues:
            return "ASE Gate 1: PASS (all ASE calls grounded in KG)"
        head = "ASE Gate 1: " + ("PASS with warnings" if self.passed else "FAIL")
        lines = [f"  [{i.severity}] {i.construct}: {i.message}"
                 + (f"  -> did you mean '{i.suggestion}'?" if i.suggestion else "")
                 for i in self.issues]
        return "\n".join([head, *lines])


def _dotted(node: ast.AST) -> list[str] | None:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return list(reversed(parts))
    return None


def referenced_entries(code: str, kg: ASEKnowledgeGraph) -> list:
    """KG entries for everything `code` imports from covered packages (evidence packs).

    Grounding rule: a snippet's evidence is the REAL introspected signature of each
    function it will call — retrieved from the KG, never quoted from an LLM.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    quals: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module \
                and node.module.startswith(tuple(kg.prefixes)):
            quals += [f"{node.module}.{a.name}" for a in node.names]
    seen: list = []
    for q in quals:
        e = kg.get(q)
        if e is not None and e not in seen:
            seen.append(e)
    return seen


class ASEValidator:
    def __init__(self, kg: ASEKnowledgeGraph | None = None) -> None:
        self.kg = kg or ASEKnowledgeGraph()
        self.modules = {n for n, d in self.kg.graph.nodes(data=True)
                        if d.get("kind") == "module"}

    def _module_members(self, module: str) -> list[str]:
        depth = module.count(".") + 1
        return [q.rsplit(".", 1)[-1] for q in self.kg.entries
                if q.startswith(module + ".") and q.count(".") == depth]

    def _resolve_imports(self, tree: ast.AST) -> tuple[dict, dict]:
        prefixes = tuple(self.kg.prefixes)
        name_to_qual: dict[str, str] = {}      # local -> full callable qualname
        name_to_module: dict[str, str] = {}    # local -> module path
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    if a.name.startswith(prefixes):
                        local = a.asname or a.name.split(".")[0]
                        name_to_module[local] = a.name if a.asname else a.name.split(".")[0]
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(prefixes):
                    for a in node.names:
                        local = a.asname or a.name
                        name_to_qual[local] = f"{node.module}.{a.name}"
        return name_to_qual, name_to_module

    @staticmethod
    def _defined_names(tree: ast.AST) -> set[str]:
        defined: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                for a in node.names:
                    defined.add((a.asname or a.name).split(".")[0])
            elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
                defined.add(node.id)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                defined.add(node.name)
            elif isinstance(node, ast.arg):
                defined.add(node.arg)
        return defined

    def validate(self, code: str, allowed_names: Iterable[str] = ()) -> ValidationReport:
        """`allowed_names`: variables predefined in the exec namespace (e.g. the
        already-built constituents a relation snippet combines)."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return ValidationReport(False, [Issue("error", "<syntax>", str(e))])

        name_to_qual, name_to_module = self._resolve_imports(tree)
        issues: list[Issue] = []

        known = self._defined_names(tree) | set(allowed_names) | set(dir(builtins))
        flagged: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load) \
                    and node.id not in known and node.id not in flagged:
                flagged.add(node.id)
                issues.append(Issue("error", node.id,
                                    "undefined name — every function must be imported "
                                    "in the snippet itself"))

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            chain = _dotted(node.func)
            if chain is None:
                continue

            # resolve candidate qualname
            if len(chain) == 1:
                cand = name_to_qual.get(chain[0])
            else:
                base = chain[0]
                if base in name_to_module:
                    cand = name_to_module[base] + "." + ".".join(chain[1:])
                elif base in name_to_qual:
                    cand = name_to_qual[base] + "." + ".".join(chain[1:])
                else:
                    cand = None          # method call on a variable -> not resolvable
            if cand is None:
                continue

            entry = self.kg.get(cand)
            if entry is None:
                parent = cand.rsplit(".", 1)[0]
                if parent in self.modules:                 # module fully known -> real miss
                    sugg = get_close_matches(chain[-1], self._module_members(parent), n=1)
                    issues.append(Issue("error", cand, "unknown ASE attribute",
                                        f"{parent}.{sugg[0]}" if sugg else ""))
                continue                                    # un-introspected submodule -> skip

            self._check_call(node, entry, issues)

        passed = not any(i.severity == "error" for i in issues)
        return ValidationReport(passed, issues)

    def _check_call(self, node: ast.Call, entry, issues: list[Issue]) -> None:
        # keyword args
        if not entry.accepts_kwargs:
            valid = {p.name for p in entry.params}
            for kw in node.keywords:
                if kw.arg is None:              # **something unpacking -> can't check
                    continue
                if kw.arg not in valid:
                    sugg = get_close_matches(kw.arg, entry.keyword_names, n=1)
                    issues.append(Issue("error", f"{entry.qualname}(...{kw.arg}=...)",
                                        "unknown keyword argument", sugg[0] if sugg else ""))
        # positional count
        if not entry.accepts_varargs and not any(isinstance(a, ast.Starred) for a in node.args):
            n_pos = len(node.args)
            if n_pos > entry.max_positional:
                issues.append(Issue("error", f"{entry.qualname}()",
                                    f"too many positional args: {n_pos} > {entry.max_positional}"))
