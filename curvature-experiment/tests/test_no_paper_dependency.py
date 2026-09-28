"""The experiment code never reaches into paper/: the manuscript depends on the records, not the reverse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_experiment_code_does_not_reference_paper_dir():
    hits = [f"{p.relative_to(ROOT)}:{i}" for p in ROOT.rglob("*.py")
            if ".cache" not in p.parts and p.resolve() != Path(__file__).resolve()  # this file names the pattern
            for i, line in enumerate(p.read_text().splitlines(), 1)
            if "paper/" in line or '"paper"' in line or "'paper'" in line]
    assert hits == [], hits
