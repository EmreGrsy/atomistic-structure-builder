# Atomistic Structure Builder

A knowledge graph grounded atomistic structure builder. It turns a natural
language request, for example *"a (6,6) carbon nanotube, 10 unit cells long,
with methanol inside"*, into a viewable, downloadable molecular or
nanocomposite geometry, built with ASE, packmol and Moltemplate. The code you
see is the code that runs.

**Geometry only.** No force fields, no MD. The combined cell is a showcase,
not an equilibrated simulation cell; MD equilibration is a later pipeline
stage. See `ROADMAP.md` for what is planned.

Modeled on GENIUS (arXiv:2512.06404): knowledge graph grounding, tiered LLM
use, and gated error recovery with verifiable yes or no rewards.

The app serves its own documentation at `/app/static/index.html`, a self
building page with six cached live examples and the benchmark section
(98 of 100 full pipeline prompts pass; a self organizing map analysis of the
failure space is included).

## The KG first backbone

The knowledge graphs are the ground truth. The LLM has exactly two jobs:
**parse** the query into typed constituents, and **orchestrate** the build
from KG evidence. It never invents builders, signatures, kwargs, or `.lt`
constructs. If it is not in the registry or a knowledge graph, it does not
get generated. Full spec: `.claude/skills/backbone/SKILL.md`.

### Flow

1. **PARSE** (LLM): query to typed constituents plus relations. The parser
   also returns a clean restatement of the message, so typos never defeat
   the deterministic correction layer.
2. **RETRIEVE** (KGs, no LLM): the real syntax per constituent, introspected
   ASE and mtagent signatures, Moltemplate constructs, registry schemas.
3. **GAP CHECK** (no LLM): diff the query against what the builders actually
   require; ask the user only what is missing or ambiguous.
4. **PROPOSE** (LLM, KG evidence in prompt): the real Python snippet per
   constituent. The snippet is the artifact.
5. **VALIDATE** (Gate 1, no LLM): static check of every snippet; failures are
   fed back for a bounded retry, then the canonical registry template is used.
6. **BUILD AND SHOW** every constituent: sandboxed execution (Gate 2) and
   geometry verification (Gate 3); code plus 3D structure per constituent.
7. **SHOWCASE assembly**: constituents combined per the stated relations into
   one cell, labeled as a showcase.

### Hard rules

- KG retrieval runs before any generation prompt is built.
- proposed == validated == shown == executed, one artifact end to end.
- The registry (`mtagent/registry.py`) is the single source of truth for
  builder names, parameter schemas and canonical snippet templates; prompts,
  clarifier questions and fallbacks all derive from it.
- ASE builds the UNIT, Moltemplate assembles the STRUCTURE.
- Every failure the LLM produces repeatedly gets a deterministic code gate,
  never another prompt tweak.

### The three gates

| Gate | What | How |
|------|------|-----|
| Gate 1 | Static validation against the KGs | AST check of Python against introspected ASE and mtagent signatures (`ase_validate.py`): hallucinated names, kwargs, arity, undefined names. `validate_lt` checks `.lt` text against the manual derived Moltemplate KG. |
| Gate 2 | Clean execution | Snippets run in a sandboxed namespace with imports restricted to `ase`, `mtagent`, `numpy`, `math` (`execute.py`); `.lt` assemblies must survive `moltemplate.sh` (`runner.py`). |
| Gate 3 | Geometry verification | Deterministic checks on the built structure (`verify.verify_atoms`): atoms exist, coordinates finite, no unphysical overlaps. Runs on every constituent and on the showcase. |

## What it builds

**Builders** (registry): `nanoparticle` (Wulff, sphere or cube; single
particle or a periodic supercrystal with sc, fcc, bcc or explicit stacking
sequence packing, plus cell replication via `repeat`), `nanotube`, `molecule`
(PubChem 3D coordinates), `solvent_box` (reference density or exact count,
mixed species supported), `surface_slab` (any element plus the compound table
below, any Miller termination including vicinal and four index forms, NxM in
plane supercells) and `bulk` (periodic conventional supercells).

**Materials with curated cells:** magnetite, rutile and anatase TiO2, alpha
quartz SiO2, wurtzite ZnO, corundum Al2O3, hematite Fe2O3, MgO, NiO, NaCl,
CeO2, SrTiO3, pyrite FeS2, GaAs, GaN, 2H MoS2, plus graphene and hBN sheets.
Every cell is a literature space group setting, stoichiometry verified.

**Relations:** `inside` (packmol fill of a hollow host), `around` (solvation
by carve and insert, mixed species dissolve into the box), `coated_by`
(packmol fill of a slab cell, joint multi species fills), `on` (grid
placement of a few molecules) and `between` (liquid film sandwiched by two
slabs, same or different materials; supercells are lattice matched
automatically with a square cell preference and an epitaxial strain cap).

## Module map (`mtagent/`)

**Backbone stages:** `registry.py` (schemas plus templates), `clarify.py`
(parse and chat routing, deterministic correction hints), `ground.py`
(retrieve, gap check, finalize, lattice matching), `propose.py` (LLM snippet
with evidence, Gate 1, retries), `execute.py` (Gate 2 sandbox), `verify.py`
(Gate 3 checks), `assemble.py` (relation driven assembly).

**Knowledge graphs:** `kg.py` and `kg_data.py` (Moltemplate manual KG plus
`validate_lt`), `ase_kg.py` and `ase_validate.py` (introspected API KG plus
the AST validator).

**Builders and tools:** `nanoparticle.py`, `wulff.py`, `nanostructures.py`
(slabs, compounds, sheets, nanotubes, bulks), `solvent.py` (boxes, mixing,
solvation), `packing.py`, `pubchem.py`, `cluster.py` (supercrystals via
Moltemplate), `moltemplate_emit.py`, `runner.py`, `viewer.py` (3Dmol.js),
`llm.py` (OpenAI access, server side key only).

**App:** `app.py` (Streamlit chat UI), `scripts/make_docs_page.py` (self
building documentation with a per example structure cache),
`scripts/eval_prompts.py` (100 prompt full pipeline benchmark),
`scripts/som_analysis.py` (SOM of the benchmark results).

## Setup

```bash
conda env create -f environment.yml     # creates the mdagent env with packmol
conda activate mdagent
```

packmol comes from conda forge; everything else is pip (see
`requirements.txt` if you manage the environment yourself, and put a packmol
binary on PATH in that case).

OpenAI key (optional; a keyless fallback covers the demo path):

```bash
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# then paste your key into .streamlit/secrets.toml
```

`secrets.toml` is gitignored and read server side only; the browser never
receives the key.

## Run

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  streamlit run app.py --server.port 8501 --server.fileWatcherType none
```

Open http://localhost:8501. The documentation page is linked from the
sidebar (served from `static/`, enabled via `enableStaticServing` in
`.streamlit/config.toml`).

Headless end to end demo, no key needed:

```bash
python scripts/backbone_demo.py --no-llm
```

Tests:

```bash
python -m pytest tests/
```

Regenerate the documentation page and its live examples (results are cached
per example under `data/cache/docs_examples`, so only changed examples
rebuild):

```bash
python scripts/make_docs_page.py
```

## Deployment

The app is a single Streamlit process with no database; state lives in the
session. What a server deployment needs:

- the conda environment above (packmol and moltemplate.sh must be on PATH),
- `.streamlit/secrets.toml` with the OpenAI key (never committed),
- a headless launch, for example:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  streamlit run app.py --server.port 8501 \
  --server.fileWatcherType none --server.headless true
```

- a reverse proxy (nginx or similar) with TLS in front of the port, and
  websocket forwarding enabled (Streamlit needs it).

### Free hosting (Streamlit Community Cloud)

The repo is set up for the free tier: `packages.txt` installs the packmol
binary from apt and `requirements.txt` covers everything else (moltemplate
is pip). Steps:

1. Push this repo to GitHub.
2. On https://share.streamlit.io sign in with GitHub and create a new app
   from the repo, main file `app.py`, Python 3.11.
3. In the app settings under Secrets paste the content of your
   `.streamlit/secrets.toml` (the OPENAI_API_KEY line). Secrets never live
   in the repo.

Free tier caveats: modest CPU and memory (large builds run slower and the
packmol timeout matters), the app sleeps after inactivity and wakes on the
next visit, and the machine's disk is ephemeral, so nothing durable can be
stored on it. For heavier use, the same repo runs on any server with the
conda environment (see above) or in a Docker image.
