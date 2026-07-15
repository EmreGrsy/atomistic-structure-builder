# Roadmap

Status of the larger work items, in rough priority order. Finished work is
kept at the bottom so the history stays visible.

## In progress

- **Build small, then replicate.** Large fills (hundreds of molecules on a
  surface, dense supercrystal solvation) should be built as one small periodic
  cell and then replicated, instead of asking packmol to converge one huge
  packing. First piece shipped: a periodic supercrystal accepts a `repeat`
  parameter (NxM or NxMxK) and replicates its cell. Next pieces: replicate for
  coated slab cells, and a sliced z layer packing fallback for explicit large
  counts (design exists: split each species into batches of about 3500 atoms,
  one packmol z slice per batch in a single input file).

## Planned

- **Deployment.** Decide the target: a lab server behind a reverse proxy, a
  Docker image, or Streamlit Community Cloud (caveat: packmol is a compiled
  binary from conda forge, so plain pip deployment is not enough; a Docker
  image is the safest route). The app itself is ready: server side secrets,
  static docs serving, one process, no database.
- **Benchmark stragglers.** Two failures remain in the 100 prompt benchmark:
  an LLM unit slip (a 2 Angstrom diameter parsed from a 2 nm request, now a
  clear error rather than a silent wrong build) and one flaky packmol fill on
  a dense supercrystal solvation. Both fall away with the replicate and
  sliced packing work above.
- **Runtime material fetch (COD).** Fetch CIF cells from the Crystallography
  Open Database at runtime so unsupported materials (calcite, hydroxyapatite,
  kaolinite, MXenes, perovskites beyond SrTiO3) become buildable. Needs
  polymorph disambiguation, partial occupancy handling, and a validation gate.
- **Oxide nanoparticles beyond magnetite.** TiO2, SiO2 and ZnO exist as slabs
  and bulks; wire them into the nanoparticle (Wulff and sphere) path.
- **Register the solid interface builder.** `build_interface` (stack two
  slabs) exists in the library but is not a registry builder yet; the
  sandwich relation covers the liquid film case only.
- **Polar surface reconstructions.** Oxide slabs are plain truncations today;
  polar terminations are not charge compensated.
- **Paper.** The evaluation harness (100 prompts, staged pass rates) and the
  SOM analysis are the quantitative foundation; write up the KG grounded
  backbone against an LLM only baseline.

## Out of scope by design

- **MD equilibration and force fields.** The combined cell is a geometry
  showcase; equilibration belongs to a later pipeline stage.

## Done

- Supercrystal `repeat` (first replicate piece), 2026 07.
- Typo robust parsing via the normalized query field, 2026 07.
- Mixed solvation (extra species dissolve into the solvent box), 2026 07.
- Hetero sandwich with automatic lattice matching (square cell preference,
  docs parity), 2026 07.
- Material library: 12 compound crystals, sheets (graphene, hBN), bulk
  builder, vicinal and four index Miller surfaces, 2026 07.
- Evaluation harness 98 of 100 with SOM analysis of the failure space,
  2026 07.
- Self building documentation page with six cached live examples, 2026 07.
