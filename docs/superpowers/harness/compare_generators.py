"""Usage: python compare_generators.py <baseline_dir> <candidate_dir>
Byte-compares main.tex and table_main.txt; pixel-compares every *.png. Exit 0 on match."""
import sys, filecmp
from pathlib import Path
import numpy as np
import matplotlib.image as mpimg
a, b = Path(sys.argv[1]), Path(sys.argv[2]); bad = 0
for n in ("main.tex", "table_main.txt"):
    ok = (b / n).exists() and filecmp.cmp(a / n, b / n, shallow=False)
    print(("OK   " if ok else "DIFF ") + n); bad += not ok
for p in sorted(a.glob("*.png")):
    q = b / p.name
    ok = q.exists() and np.array_equal(mpimg.imread(p), mpimg.imread(q))
    print(("OK   " if ok else "DIFF ") + p.name); bad += not ok
print("PASS" if not bad else f"FAIL ({bad})"); sys.exit(1 if bad else 0)
