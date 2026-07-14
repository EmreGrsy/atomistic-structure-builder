"""EXECUTE — run exactly the snippet the user was shown, sandboxed (Gate 2).

The snippet executes in a namespace whose imports are restricted to the same packages
Gate 1 validated against (ase, mtagent, numpy, math) — so nothing outside the KG's
reach can run — and must bind the result to `atoms`. Gate 3 (verify.verify_atoms)
then checks the produced geometry deterministically.
"""
from __future__ import annotations

import builtins as _builtins

from ase import Atoms

_ALLOWED_ROOTS = {"ase", "mtagent", "numpy", "math"}

_SAFE_NAMES = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int",
    "isinstance", "len", "list", "max", "min", "print", "range", "repr", "round",
    "set", "sorted", "str", "sum", "tuple", "zip",
    "ArithmeticError", "AttributeError", "Exception", "ImportError", "IndexError",
    "KeyError", "RuntimeError", "TypeError", "ValueError", "ZeroDivisionError",
)


def _guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name.split(".")[0] not in _ALLOWED_ROOTS:
        raise ImportError(f"snippets may only import from {sorted(_ALLOWED_ROOTS)} "
                          f"(tried '{name}')")
    return _builtins.__import__(name, globals, locals, fromlist, level)


def run_snippet(code: str, inputs: dict | None = None) -> Atoms:
    """Execute a validated snippet; return the `atoms` it defines. Raises on any failure."""
    ns: dict = {"__builtins__": {**{n: getattr(_builtins, n) for n in _SAFE_NAMES},
                                 "__import__": _guarded_import},
                **(inputs or {})}
    exec(compile(code, "<snippet>", "exec"), ns)
    atoms = ns.get("atoms")
    if not isinstance(atoms, Atoms):
        raise RuntimeError("snippet did not produce an ase.Atoms object named 'atoms'")
    return atoms
