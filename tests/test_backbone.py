"""Backbone regression tests — templates are KG-grounded, gaps are registry-derived,
snippets execute sandboxed, and the keyless path walks end to end."""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent import clarify, ground
from mtagent.assemble import RELATIONS, combine_template, fill_inside
from mtagent.execute import run_snippet
from mtagent.packing import find_packmol
from mtagent.registry import BUILDERS
from mtagent.verify import verify_atoms

SAMPLE_SPECS = {
    "nanoparticle": [
        {"material": "magnetite", "diameter": 30, "shape": "wulff",
         "hydroxylate": True, "n_particles": 1},
        {"material": "gold", "diameter": 20, "shape": "cube", "n_particles": 1},
        {"material": "magnetite", "diameter": 20, "shape": "sphere", "n_particles": 2},
    ],
    "nanotube": [{"n": 6, "m": 6, "length": 5}],
    "molecule": [{"name": "methanol"}],
    "solvent_box": [{"molecule": "water", "box_size": 30}],
    "surface_slab": [{"element": "Au", "miller": "111"}],
}


@pytest.fixture(autouse=True)
def keyless(monkeypatch):
    """Tests exercise the deterministic path — never call OpenAI."""
    monkeypatch.setattr(clarify, "have_openai_key", lambda: False)


# ----------------------------- Gate 1 grounding --------------------------------

def test_every_builder_template_passes_gate1():
    v = ground.validator()
    for name, specs in SAMPLE_SPECS.items():
        for spec in specs:
            code = BUILDERS[name].template(BUILDERS[name].defaults(spec))
            rep = v.validate(code, allowed_names=("host", "guest"))
            assert rep.passed, f"{name}{spec}:\n{rep.summary()}\n{code}"


def test_every_combine_template_passes_gate1():
    v = ground.validator()
    for kind in RELATIONS:
        for guest_builder in ("molecule", "solvent_box"):
            code = combine_template(kind, "host", "guest", guest_builder,
                                    {"count": 5, "box_size": 40})
            rep = v.validate(code, allowed_names=("host", "guest"))
            assert rep.passed, f"{kind}/{guest_builder}:\n{rep.summary()}"


def test_validator_rejects_hallucinated_kwarg():
    rep = ground.validator().validate(
        "from mtagent.nanostructures import build_nanotube\n"
        "atoms = build_nanotube(n=6, m=6, lenght=5)")
    assert not rep.passed


# ------------------------- parse → gaps → finalize -----------------------------

def test_keyword_parse_cnt_methanol_inside():
    st = clarify.parse_query("a carbon nanotube with methanol inside")
    builders = {c["builder"] for c in st["constituents"]}
    assert builders == {"nanotube", "molecule"}
    assert st["relation"] and st["relation"]["kind"] == "inside"


def test_gaps_are_registry_derived_and_close():
    st = clarify.parse_query("a (6,6) carbon nanotube with methanol inside")
    for _ in range(10):
        gs = ground.gaps(st)
        if not gs:
            break
        st = clarify.apply_answer(st, "default", gs[0])
    assert not ground.gaps(st)
    final = ground.finalize(st)
    cnt = next(c for c in final["constituents"] if c["builder"] == "nanotube")
    assert cnt["spec"]["n"] == 6 and cnt["spec"]["length"] == 10


def test_no_relation_question_for_single_constituent():
    st = clarify.parse_query("a gold nanoparticle of diameter 3 nm")
    assert all(g.target != "relation" for g in ground.gaps(st))
    np_spec = st["constituents"][0]["spec"]
    assert np_spec["diameter"] == pytest.approx(30.0)


def test_solvent_box_must_encompass_solute():
    st = clarify.parse_query("a 4 nm magnetite nanoparticle in water")
    for _ in range(10):
        gs = ground.gaps(st)
        if not gs:
            break
        st = clarify.apply_answer(st, "default", gs[0])
    final = ground.finalize(st)
    box = next(c for c in final["constituents"] if c["builder"] == "solvent_box")
    assert box["spec"]["box_size"] >= 60.0


# ----------------------------- Gate 2 sandbox ----------------------------------

def test_sandbox_blocks_foreign_imports():
    with pytest.raises(ImportError):
        run_snippet("import os\natoms = None")


def test_snippet_must_produce_atoms():
    with pytest.raises(RuntimeError):
        run_snippet("x = 1")


def test_execute_nanotube_template_and_gate3():
    code = BUILDERS["nanotube"].template({"n": 6, "m": 6, "length": 3})
    atoms = run_snippet(code)
    report = verify_atoms(atoms)
    assert report.passed, report.summary()
    assert len(atoms) == 72


# --------------------------- assembly (external tools) -------------------------

@pytest.mark.skipif(find_packmol() is None, reason="packmol not installed")
def test_fill_inside_tube():
    from ase import Atoms
    host = run_snippet(BUILDERS["nanotube"].template({"n": 8, "m": 8, "length": 6}))
    guest = Atoms("Ne", positions=[[0, 0, 0]])
    filled = fill_inside(host, guest, n=4)
    assert len(filled) == len(host) + 4
    assert verify_atoms(filled).passed


def test_mof_cell_is_the_published_framework():
    """The bundled CIF is ground truth: if it drifts, every MOF build is wrong."""
    atoms = run_snippet(BUILDERS["mof"].template({"name": "ZIF-8"}))
    assert atoms.get_chemical_formula() == "C96H120N48Zn12"   # Park, PNAS 2006
    assert len(atoms) == 276 and all(atoms.get_pbc())
    assert abs(atoms.cell.lengths()[0] - 16.991) < 0.01
    assert verify_atoms(atoms).passed


def test_mof_slab_and_nanoparticle_are_refused():
    """Cutting a MOF would slice its linkers — say so, do not build a wreck."""
    for builder, spec in (("surface_slab", {"element": "ZIF-8", "miller": "110"}),
                          ("nanoparticle", {"material": "zif-8", "diameter": 40,
                                            "shape": "sphere"})):
        with pytest.raises(ValueError, match="metal-organic framework"):
            BUILDERS[builder].template(spec)


@pytest.mark.skipif(find_packmol() is None, reason="packmol not installed")
def test_fill_pores_puts_guests_in_the_pores():
    from mtagent.assemble import fill_pores
    from ase import Atoms

    host = run_snippet(BUILDERS["mof"].template({"name": "ZIF-8"}))
    guest = Atoms("Ne", positions=[[0, 0, 0]])
    loaded = fill_pores(host, guest, n=8)
    assert len(loaded) == len(host) + 8
    assert (loaded.cell.lengths() == host.cell.lengths()).all()   # cell kept
    d = loaded.get_all_distances(mic=True)
    # guests clear the framework AND its periodic images, or they are in a wall
    assert d[len(host):, :len(host)].min() > 1.9
    assert verify_atoms(loaded).passed


@pytest.mark.skipif(shutil.which("moltemplate.sh") is None,
                    reason="moltemplate not installed")
def test_cluster_via_moltemplate_gate1_gate2():
    code = BUILDERS["nanoparticle"].template(
        {"material": "gold", "diameter": 12, "shape": "sphere", "n_particles": 2})
    atoms = run_snippet(code)
    assert atoms.info["cluster"]["n_units"] == 2
    assert verify_atoms(atoms).passed
