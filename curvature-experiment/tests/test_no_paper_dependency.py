"""The experiment code never reaches into paper/: the manuscript depends on the records, not the reverse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# test_sweep_extract.py reads the committed main.tex read-only as the oracle for the ported record readers
# (it pins sweep/extract.py to the cells the manuscript prints); no experiment code imports or writes paper/.
ALLOWED = {ROOT / "tests" / "test_sweep_extract.py"}


def test_experiment_code_does_not_reference_paper_dir():
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in ROOT.rglob("*.py")
            if ".cache" not in p.parts and p.resolve() != Path(__file__).resolve()  # this file names the pattern
            and p.resolve() not in ALLOWED
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "paper/" in line or '"paper"' in line or "'paper'" in line]
    assert hits == [], hits
