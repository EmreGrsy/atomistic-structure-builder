"""KG coverage test — the objective completeness metric.

Extract every construct the official manual actually uses (data sections in
write()/write_once(), and transform/method calls), then check them against the KG.
A complete KG should recognize all real constructs (doc metavariables excluded).
This is how we know the KG covers the manual, not just our own emitter.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mtagent.kg import MoltemplateKG
from mtagent import kg_data as D

MANUAL = Path("data/refs/manual.txt")

# Doc metavariables / obvious extraction noise to ignore (not real constructs).
# "Data Foo Fee Fum" is the manual's own placeholder for a *custom* section (§5.6).
SECTION_NOISE = {"FileName", "file", "Data Foo Fee Fum", "filename"}
CUSTOM_PLACEHOLDER = {"Data Foo Fee Fum"}
METHOD_NOISE = D.PLACEHOLDERS | {"py", "sh", "gz", "dat", "txt", "in", "out",
                                 "e", "g", "i", "lt", "pdb", "xyz", "data"}

kg = MoltemplateKG()
text = MANUAL.read_text()

# --- sections used in the manual ---------------------------------------------
raw_sections = set(re.findall(r'write(?:_once)?\s*\(\s*[\'"]([^\'"]+)[\'"]', text))
# normalize: strip a trailing " (...)" filename artifact from PDF extraction
sections = {s.split(" (")[0].strip() for s in raw_sections if "\n" not in s}
sections = {s for s in sections if s.startswith(("Data ", "In "))       # only LAMMPS sections
            and s not in CUSTOM_PLACEHOLDER}                            # drop the §5.6 demo name
known_sec = {s for s in sections if s in D.DATA_SECTIONS}
unknown_sec = sections - known_sec

# --- transform / method calls in the manual ----------------------------------
raw_methods = set(re.findall(r'\.([a-zA-Z_]\w*)\s*\(', text))
# exclude noise, doc metavariables, and ALL-CAPS user macro placeholders (e.g. XFORMS3)
methods = {m for m in raw_methods if m not in METHOD_NOISE and len(m) > 1
           and not m.isupper()}
valid_methods = set(D.TRANSFORMS) | D.FORMAT_METHODS
known_m = {m for m in methods if m in valid_methods}
unknown_m = methods - known_m

print("=== SECTIONS (LAMMPS 'Data '/'In ') used in manual ===")
print(f"  recognized : {len(known_sec)}/{len(sections)}")
if unknown_sec:
    print(f"  UNKNOWN    : {sorted(unknown_sec)}")

print("\n=== TRANSFORM/METHOD calls used in manual ===")
print(f"  recognized : {sorted(known_m)}")
if unknown_m:
    print(f"  UNKNOWN    : {sorted(unknown_m)}")

total = len(sections) + len(methods)
known = len(known_sec) + len(known_m)
print(f"\nKG coverage of manual constructs: {known}/{total} "
      f"({100*known/total:.0f}%)  [doc metavariables excluded]")
