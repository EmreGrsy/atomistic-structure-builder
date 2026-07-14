# Moltemplate Agent

An AI agent that turns a natural-language request (e.g. *"a (6,6) carbon nanotube,
10 unit cells long, with methanol inside"*) into a **viewable, downloadable
molecular / nanocomposite geometry**, built with ASE + packmol + Moltemplate.

**Geometry only** — no force fields, no MD. The combined cell is a *showcase*,
not an equilibrated simulation cell (MD equilibration is a later pipeline stage).

Modeled on **GENIUS** (arXiv:2512.06404): knowledge-graph grounding + tiered LLM
use + gated error recovery with verifiable yes/no rewards.

## The KG-first backbone

The knowledge graphs are the ground truth. The LLM has exactly two jobs:
**parse** the query into typed constituents, and **orchestrate** the build from
KG evidence. It never invents builders, signatures, kwargs, or `.lt` constructs —
if it's not in the registry or a KG, it doesn't get generated.

Full spec: `.claude/skills/backbone/SKILL.md`.

### Flow (7 steps)

1. **PARSE** (LLM) — query → typed constituents + relations
   (`inside` / `around` / `coated_by` / `on`).
2. **RETRIEVE** (KGs, no LLM) — pull the real syntax per constituent: introspected
   ASE/mtagent signatures, Moltemplate constructs, registry param schemas.
3. **GAP CHECK** (no LLM) — diff the query against what the builders actually
   require; ask the user **only** what is missing or ambiguous.
4. **PROPOSE** (LLM, KG evidence in-prompt) — write the real Python snippet per
   constituent. The snippet **is** the artifact.
5. **VALIDATE** (Gate 1, no LLM) — static check of every proposed snippet; on
   failure the issues are fed back for a bounded retry (3x), then the canonical
   registry template is used (KG-grounded by construction).
6. **BUILD & SHOW** every constituent — sandboxed execution (Gate 2) + geometry
   verification (Gate 3); code + 3D structure shown per constituent.
7. **SHOWCASE assembly** — combine constituents per the stated relations into one
   cell, labeled as a showcase.

### Hard rules

- KG retrieval runs **before** any generation prompt is built.
- **proposed == validated == shown == executed** — one artifact end to end,
  byte for byte.
- The **registry** (`mtagent/registry.py`) is the single source of truth for
  builder names, param schemas, and canonical snippet templates; prompts,
  clarifier questions, and fallbacks all derive from it.
- **ASE builds the UNIT, Moltemplate assembles the STRUCTURE.**

### The three gates

| Gate | What | How |
|------|------|-----|
| **Gate 1** | Static validation vs KGs | AST check of Python against introspected ASE/mtagent signatures (`ase_validate.py`): hallucinated names, kwargs, arity — each with a suggestion. `validate_lt` checks `.lt` text against the manual-derived Moltemplate KG. |
| **Gate 2** | Clean execution | Snippets run in a sandboxed namespace with imports restricted to `ase` / `mtagent` / `numpy` / `math` (`execute.py`); `.lt` assemblies must survive `moltemplate.sh` (`runner.py`). |
| **Gate 3** | Geometry verification | Deterministic checks on the built structure (`verify.verify_atoms`): atoms exist, coordinates finite, no non-physical overlaps. Runs on every constituent and on the showcase. |

## Module map (`mtagent/`)

**Backbone stages**

- `registry.py` — single source of truth: five builders (`nanoparticle`,
  `nanotube`, `molecule`, `solvent_box`, `surface_slab`) with param schemas
  (including the question to ask when a value is missing) and canonical snippet
  templates calling real functions.
- `clarify.py` — PARSE: `parse_query` (query → constituents + relation,
  constrained to the registry catalog) and `apply_answer` (merge a reply into the
  spec). A keyword parser keeps the demo path alive without an OpenAI key.
- `ground.py` — RETRIEVE + GAP CHECK: builds the KGs once per process, assembles
  evidence packs per constituent, computes `gaps()` from the registry schemas,
  `finalize()` completes the spec.
- `propose.py` — PROPOSE: the LLM writes the snippet with KG evidence in-prompt;
  Gate 1 validates; bounded retries; falls back to the canonical template.
- `execute.py` — Gate 2: sandboxed `run_snippet` (restricted imports/builtins),
  must bind an `ase.Atoms` to `atoms`.
- `verify.py` — Gate 3: deterministic geometry checks (KD-tree clash detection).
- `assemble.py` — relation-driven assembly (`inside`, `around`, `coated_by`,
  `on`) into one showcase cell.
- `app.py` (repo root) — Streamlit chat UI: shows the Gate-1-validated snippet
  for every constituent up front, takes plain-language adjustments, builds on
  request, renders 3D views and downloads.

**Knowledge graphs**

- `kg.py` + `kg_data.py` — Moltemplate KG extracted from the official manual
  (`data/refs/moltemplate_manual.pdf`); keyword + graph retrieval and the
  `validate_lt` Gate 1 validator.
- `ase_kg.py` + `ase_validate.py` — API KG built by **introspecting** installed
  `ase` + `mtagent` (version-matched real signatures) and the AST-based Gate 1
  for Python snippets.

**Builders and tools**

- `nanoparticle.py` — spherical FCC metal nanoparticle carve.
- `wulff.py` — multi-element Wulff-construction carve for cubic crystals
  (e.g. magnetite spinel).
- `hydroxylate.py` — coordination-based surface hydroxylation of oxide particles.
- `nanostructures.py` — nanotubes, generic spherical NPs, surface slabs, interfaces.
- `solvent.py` — solvent box at reference density + carve-and-insert solvation.
- `packing.py` — clash-free shell packing via packmol.
- `pubchem.py` — real 3D molecule coordinates from PubChem, provenance-tracked.
- `cluster.py` — multi-NP cluster assembly via Moltemplate (deterministic lattice).
- `moltemplate_emit.py` — emit `.lt` from packed geometry (molecule defined once,
  instantiated N times with Kabsch-recovered `.rot().move()` transforms).
- `runner.py` — run `moltemplate.sh` (Gate 2 for `.lt`) and convert LAMMPS data
  back to `.xyz`.
- `viewer.py` — 3Dmol.js HTML viewer (embedded in Streamlit).
- `llm.py` — OpenAI access; key read server-side from env or Streamlit secrets,
  `have_openai_key()` gates the keyless fallback.

## Setup & run

Environment: conda env **`mdagent`** (ASE, moltemplate, packmol, streamlit,
openai, scipy; see `requirements.txt`).

OpenAI key (optional — a keyless fallback covers the demo path): copy
`.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and set

```toml
OPENAI_API_KEY = "sk-..."
```

Launch the app (from the `mdagent` env):

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  streamlit run app.py --server.port 8501 --server.fileWatcherType none
```

Headless end-to-end demo (no LLM key needed):

```bash
python scripts/backbone_demo.py --no-llm
```

Tests:

```bash
python -m pytest tests/    # test_backbone.py, test_kg.py, test_ase_kg.py
```

Other demos: `scripts/kg_demo.py`, `scripts/ase_kg_demo.py`,
`scripts/wulff_demo.py`, `scripts/hydroxylate_demo.py`,
`scripts/kg_coverage.py` (manual-coverage check for the Moltemplate KG).
