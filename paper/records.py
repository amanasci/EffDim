"""Where the paper's generators find the experiment records. The only link from paper/ to the code."""
import os
from pathlib import Path

PAPER = Path(__file__).resolve().parent
RECORDS = Path(os.environ.get("EFFDIM_CACHE_DIR") or PAPER.parent / "curvature-experiment" / ".cache").resolve()
