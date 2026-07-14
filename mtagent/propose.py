"""PROPOSE — the LLM writes the actual snippet, with KG evidence in the prompt.

The snippet IS the artifact: what is proposed here is what Gate 1 validates, what the
user sees, and what executes — byte for byte. The LLM is constrained to the functions
whose real signatures are in the evidence pack; Gate 1 (AST vs introspected
signatures) rejects anything else and the issues are fed back for a bounded retry.
Without an OpenAI key (or if the LLM keeps failing Gate 1) the canonical registry
snippet is used — it is KG-grounded by construction.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .ase_validate import ValidationReport
from .ground import Evidence, validator
from .llm import DEFAULT_MODEL, chat_text, have_openai_key

_SYSTEM = """You write SHORT Python snippets that build molecular geometries.
Hard rules:
- Call ONLY the functions whose exact signatures are provided. No other imports, no
  other functions, no invented keyword arguments.
- IMPORT every function you call in the snippet itself (`from <module> import <name>`,
  module as shown in the signature) — the snippet runs in a bare namespace.
- The snippet must end with the final structure bound to the variable `atoms`
  (an ase.Atoms object). Input structures mentioned in the task are already defined
  as variables — do not rebuild them.
- No file I/O, no comments, no prints, no functions/classes — plain statements only.
Return ONLY the code (no markdown fences, no explanation)."""


@dataclass
class Proposal:
    target: str
    code: str
    source: str                     # "llm" | "template"
    report: ValidationReport
    attempts: int = 0
    note: str = ""


def _strip_fences(text: str) -> str:
    m = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL)
    return (m.group(1) if m else text).strip()


def propose(evidence: Evidence, task: str, model: str = DEFAULT_MODEL,
            max_attempts: int = 3) -> Proposal:
    """Return a Gate-1-validated snippet for `task`, grounded in `evidence`."""
    v = validator()
    allowed = evidence.variables
    if not have_openai_key():
        return Proposal(evidence.target, evidence.template, "template",
                        v.validate(evidence.template, allowed_names=allowed),
                        note="no OpenAI key — canonical registry snippet")

    messages = [{"role": "user", "content": f"Task: {task}\n\n{evidence.as_prompt()}"}]
    for attempt in range(1, max_attempts + 1):
        code = _strip_fences(chat_text(messages, _SYSTEM, model=model))
        report = v.validate(code, allowed_names=allowed)
        if report.passed:
            return Proposal(evidence.target, code, "llm", report, attempt)
        messages += [{"role": "assistant", "content": code},
                     {"role": "user", "content":
                      f"Gate 1 rejected that snippet:\n{report.summary()}\n"
                      "Fix ONLY these issues and return the corrected code."}]
    return Proposal(evidence.target, evidence.template, "template",
                    v.validate(evidence.template, allowed_names=allowed), max_attempts,
                    note=f"LLM snippet failed Gate 1 {max_attempts}x — "
                         "canonical registry snippet used")
