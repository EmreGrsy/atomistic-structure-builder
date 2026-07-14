# Evaluation prompt set (100 prompts)

| # | category | complexity | prompt | outcome |
|---|----------|------------|--------|---------|
| 1 | metal np | basic | a 3 nm gold nanoparticle | pass |
| 2 | metal np | basic | silver nanoparticle, 25 angstrom diameter | pass |
| 3 | metal np | basic | a 2 nm platinum nanocube | pass |
| 4 | metal np | basic | copper nanoparticle | pass |
| 5 | metal np | basic | a palladium nanoparticle of 30 A | pass |
| 6 | metal np | basic | nickel nanoparticle 2.5 nm | pass |
| 7 | metal np | basic | a titanium nanoparticle, 24 angstrom | pass |
| 8 | metal np | basic | aluminum nanoparticle with diameter 20 A | fail at build |
| 9 | metal np | basic | a 2.8 nm silver cube | pass |
| 10 | metal np | basic | gold nanoparticle, sphere, 18 angstrom | pass |
| 11 | oxide np | basic | a 3 nm magnetite nanoparticle | pass |
| 12 | oxide np | standard | magnetite nanoparticle, wulff shape, 2.5 nm | pass |
| 13 | oxide np | complex | a magnetite nanoparticle with only 111 and 100 facets, 3 nm | pass |
| 14 | oxide np | basic | magnetite nanosphere of 28 angstrom | pass |
| 15 | oxide np | basic | a 2 nm magnetite cube | pass |
| 16 | oxide np | basic | iron oxide nanoparticle 3 nm | pass |
| 17 | oxide np | standard | Fe3O4 nanoparticle, 26 A, wulff | pass |
| 18 | oxide np | standard | magnetite particle where the 110 surface energy is 1.2, 3 nm | pass |
| 19 | oxide np | standard | a faceted magnetite nanoparticle with gamma_100 of 1.1, 30 A | pass |
| 20 | oxide np | standard | magnetite nanoparticle 2 nm with no 110 facet | pass |
| 21 | solvation | standard | a 2 nm magnetite nanoparticle in water | pass |
| 22 | solvation | standard | gold nanoparticle, 2 nm, solvated in ethanol | pass |
| 23 | solvation | standard | a 20 A silver nanoparticle in a water box | pass |
| 24 | solvation | standard | magnetite nanoparticle 2.5 nm surrounded by methanol | pass |
| 25 | solvation | standard | a copper nanoparticle, 18 A, in water | pass |
| 26 | solvation | standard | 2 nm platinum nanoparticle immersed in water | pass |
| 27 | solvation | standard | a small magnetite cube in ethanol, 20 A | pass |
| 28 | solvation | standard | nickel nanoparticle 2 nm inside a box of water | pass |
| 29 | solvation | standard | a 25 A gold sphere in toluene | pass |
| 30 | solvation | standard | magnetite wulff nanoparticle, 24 A, in hexane | pass |
| 31 | element slab | basic | a gold 111 surface | pass |
| 32 | element slab | basic | aluminum 001 slab | pass |
| 33 | element slab | standard | copper 110 surface, 3x3 | pass |
| 34 | element slab | basic | an iron 110 slab | pass |
| 35 | element slab | basic | platinum 557 surface | pass |
| 36 | element slab | basic | titanium 0001 surface | pass |
| 37 | element slab | basic | a silicon 100 slab | pass |
| 38 | element slab | basic | germanium 100 surface | pass |
| 39 | element slab | standard | a tungsten 110 surface, 4x4 supercell | pass |
| 40 | element slab | standard | silver 100 slab with 12 angstrom vacuum | pass |
| 41 | compound slab | standard | a rutile TiO2 110 surface | pass |
| 42 | compound slab | standard | anatase 101 slab | pass |
| 43 | compound slab | standard | alpha quartz 0001 surface | pass |
| 44 | compound slab | basic | a zinc oxide 0001 surface | pass |
| 45 | compound slab | basic | GaAs 110 slab | pass |
| 46 | compound slab | basic | an MgO 100 surface | pass |
| 47 | compound slab | standard | SrTiO3 100 slab | pass |
| 48 | compound slab | standard | sodium chloride 100 surface | pass |
| 49 | compound slab | basic | an alumina 0001 slab | pass |
| 50 | compound slab | standard | hematite 0001 surface | pass |
| 51 | interface | standard | water on an aluminum 111 surface | pass |
| 52 | interface | standard | 30 water molecules on a gold 111 slab | pass |
| 53 | interface | standard | ethanol on a magnetite 001 surface | pass |
| 54 | interface | standard | a silicon 100 surface covered with water | pass |
| 55 | interface | standard | 15 methanol molecules on a platinum 111 surface | pass |
| 56 | interface | standard | water on rutile TiO2 110 | pass |
| 57 | interface | standard | 10 oleic acid molecules on a gold 111 slab | pass |
| 58 | interface | standard | a copper 111 surface coated with 40 waters | pass |
| 59 | interface | standard | ethanol layer on quartz 0001 | pass |
| 60 | interface | standard | 2 water molecules placed on an MgO 100 surface | pass |
| 61 | sandwich | complex | water between two gold 111 slabs | pass |
| 62 | sandwich | complex | 30 water molecules sandwiched between two magnetite 001 slabs | pass |
| 63 | sandwich | complex | water between two graphene sheets | pass |
| 64 | sandwich | complex | ethanol confined between two aluminum 111 slabs | pass |
| 65 | sandwich | complex | water between a magnetite 001 slab and a rutile 110 slab | pass |
| 66 | sandwich | complex | a water film between two silicon 100 surfaces | pass |
| 67 | sandwich | complex | 40 waters between two copper 111 slabs | pass |
| 68 | sandwich | complex | methanol between two graphene layers | pass |
| 69 | sandwich | complex | water sandwiched between MgO 100 and NaCl 100 slabs | pass |
| 70 | sandwich | complex | 20 ethanol molecules between two quartz 0001 slabs | pass |
| 71 | supercrystal | complex | a supercrystal of 4 magnetite nanoparticles, 2 nm each | fail at assemble |
| 72 | supercrystal | complex | FCC supercrystal of 4 gold nanoparticles, 18 A | pass |
| 73 | supercrystal | standard | a cluster of 8 magnetite nanoparticles, 2 nm | pass |
| 74 | supercrystal | complex | bcc superlattice of 2 magnetite nanoparticles, 20 A | pass |
| 75 | supercrystal | complex | magnetite supercrystal made of magnetite nanoparticles, use FCC | pass |
| 76 | supercrystal | complex | a cluster of 3 silver nanoparticles with 15 A gaps, 2 nm each | pass |
| 77 | supercrystal | standard | 4 copper nanoparticles arranged in a cluster, 18 A each | pass |
| 78 | supercrystal | complex | fcc superlattice of 4 magnetite cubes, 18 A | pass |
| 79 | supercrystal | standard | a dimer of two 2 nm gold nanoparticles | pass |
| 80 | supercrystal | complex | supercrystal with 6 magnetite nanoparticles, 20 A diameter | pass |
| 81 | nanotube | standard | a (6,6) carbon nanotube, 10 cells long | pass |
| 82 | nanotube | complex | carbon nanotube with methanol inside | pass |
| 83 | nanotube | complex | a (10,10) CNT filled with 8 ethanol molecules | pass |
| 84 | nanotube | standard | a zigzag (9,0) carbon nanotube | pass |
| 85 | nanotube | standard | water inside a (8,8) carbon nanotube | pass |
| 86 | nanotube | complex | a carbon nanotube, 12 unit cells, with 5 waters inside | pass |
| 87 | nanotube | complex | (7,7) CNT with acetone inside | pass |
| 88 | nanotube | standard | a short (6,6) nanotube, 6 cells | pass |
| 89 | nanotube | standard | carbon nanotube containing 10 methanol molecules | pass |
| 90 | nanotube | complex | a (12,0) carbon nanotube with water inside | pass |
| 91 | other | standard | magnetite bulk, 2x2x2 | pass |
| 92 | other | standard | gold bulk crystal 3x3x3 | pass |
| 93 | other | basic | bulk silicon | pass |
| 94 | other | basic | an MgO bulk crystal | pass |
| 95 | other | basic | a single oleic acid molecule | pass |
| 96 | other | basic | a caffeine molecule | pass |
| 97 | other | basic | a 25 angstrom box of water | pass |
| 98 | other | standard | an ethanol solvent box, 30 A | pass |
| 99 | other | basic | a graphene sheet | pass |
| 100 | other | basic | an h-BN sheet | pass |

**98/100 pass the full pipeline** (parse, specification, static validation, sandboxed build, geometry checks, assembly).
Complexity distribution (gpt-4o-mini scored): basic 32.0%, standard 45.0%, complex 23.0%