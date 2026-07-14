---
name: backbone
description: Moltemplate Agent backbone — KG retrieval first, LLM only parses the query and orchestrates. Read before touching any pipeline stage, prompt, agent, or gate.
---

# Backbone

The KGs (`mtagent/ase_kg.py`, `mtagent/kg.py`) are the ground truth. The LLM has
two jobs only: **parse** the query into constituents, and **orchestrate** the
build from KG evidence. It never invents builders, signatures, kwargs, or `.lt`
constructs — if it's not in the registry or a KG, it doesn't get generated.

## Flow

1. **PARSE** (LLM): query → typed constituents + relations between them.
2. **RETRIEVE** (KGs, no LLM): for each constituent, pull the real syntax —
   ASE signatures, Moltemplate constructs, registry param schemas.
3. **GAP CHECK** (no LLM): diff the query against what the retrieved builders
   actually require. Ask the user **only** what's missing or ambiguous
   (params, and relations like inside/around/coated/on). Never a scripted
   question list.
4. **PROPOSE** (LLM, KG evidence in-prompt): write the real snippet per
   constituent. The snippet **is** the artifact — no post-hoc script strings.
5. **VALIDATE** (Gate 1, no LLM): `ase_validate` on Python, `validate_lt` on
   `.lt`. Fail → regenerate with the issues fed back, bounded retries.
6. **BUILD & SHOW every constituent**: execute the validated snippets
   (Gate 2 clean run, Gate 3 verify); show code + 3D result per constituent.
7. **SHOWCASE assembly**: combine constituents per the stated relations into
   one cell and show it — labeled as a *showcase*, not a simulation-ready
   cell. The true final cell needs MD equilibration, which is **out of scope**
   for the Moltemplate agent (later pipeline stage).

## Worked example — "carbon nanotube, methanol"

- PARSE → constituents: `nanotube(C)`, `molecule(methanol)`; relation unknown.
- RETRIEVE → ASE KG: `ase.build.nanotube(n, m, length, bond, symbol, ...)`
  real signature; methanol via molecule builder; Moltemplate KG: constructs
  allowed for assembling them into one system.
- GAP CHECK → missing: `(n, m)`, length, methanol count, and the relation →
  ask: "Should the methanol be **inside** the nanotube or around it?"
- PROPOSE → one snippet per constituent using exactly the retrieved syntax.
- VALIDATE → Gate 1; then BUILD & SHOW the CNT and the methanol separately
  (code + structure), then the combined showcase cell (methanol placed inside,
  if that was the answer).

## Hard rules

- KG retrieval runs **before** any generation prompt is built.
- proposed == validated == shown == executed — one artifact end to end.
- Registry (`mtagent/agents/`) is the single source of truth for builder
  names, param schemas, snippets; prompts and clarifiers derive from it.
- ASE builds the UNIT, Moltemplate assembles the STRUCTURE.
- One path — `decompose.py`, `llm_decompose.py`, `suggest.py` are legacy;
  fold in or retire, don't extend.
- MD/equilibration is out of scope; the combined cell is a showcase only.
