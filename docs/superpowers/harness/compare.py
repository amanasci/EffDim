"""Compare two smoke-output directories record by record.

Usage: python compare.py <baseline_dir> <candidate_dir>

Exit 0 when every *.jsonl in the baseline has an identical counterpart in the candidate
after dropping volatile keys (timestamps, wallclock, versions, repo/colleague heads, paths)
and colleague material (keys matching COLLEAGUE_KEY; rows whose instrument/estimator names
the colleague). Floats compare exactly, NaN equal to NaN. *.npz arrays compare with
np.array_equal(equal_nan=True). Prints the first differing path per file.
"""
import json, math, re, sys
from pathlib import Path
import numpy as np

VOLATILE = re.compile(r"(timestamp|wallclock|^python$|^torch$|^numpy$|version|repo_head|git_head|hostname|path|record_path|output_root|argv|elapsed|_root$|^device$|^deterministic$|^gpu_name$|^cuda_version$)", re.I)
# Task 4 (Stage 3) widening: "topology_is_shim" is colleague-checkout provenance (whether the
# colleague's topology import came from the read-only shim) with no "colleague"/"col" substring;
# it only ever appeared in the pre-removal baseline's environment row. Proof that this drops
# nothing else is in commit f235f49's message.
COLLEAGUE_KEY = re.compile(r"(colleague|(^|_)col($|_)|^topology_is_shim$)")
# Task 4 (Stage 3) widening: rows for the colleague's "K_H_cross" instrument were named
# "his_K_H_cross" (never containing the literal substring "colleague"). Matched as a whole
# underscore-delimited token so "physics"/"this" etc. are not affected. Proof that this drops
# nothing else is in commit f235f49's message.
HIS_ROW = re.compile(r"(^|_)his($|_)", re.I)

def is_colleague_row(r):
    return any(isinstance(v, str) and ("colleague" in v.lower() or HIS_ROW.search(v))
               for k, v in r.items() if k in ("instrument", "estimator", "method", "arm", "row", "column"))

def clean(o):
    if isinstance(o, dict):
        return {k: clean(v) for k, v in o.items() if not VOLATILE.search(k) and not COLLEAGUE_KEY.search(k)}
    if isinstance(o, list):
        return [clean(v) for v in o]
    return o

def first_diff(a, b, p=""):
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return f"{p}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{p}.{k}: missing on {'baseline' if k not in a else 'candidate'}"
            d = first_diff(a[k], b[k], f"{p}.{k}")
            if d: return d
        return None
    if isinstance(a, list):
        if len(a) != len(b): return f"{p}: len {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"{p}[{i}]")
            if d: return d
        return None
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return None
    return None if a == b else f"{p}: {a!r} != {b!r}"

def load(path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return [clean(r) for r in rows if not is_colleague_row(r)]

def main(base, cand):
    base, cand = Path(base), Path(cand)
    bad = 0
    for bf in sorted(base.glob("*.jsonl")):
        cf = cand / bf.name
        if not cf.exists():
            print(f"MISSING {bf.name}"); bad += 1; continue
        a, b = load(bf), load(cf)
        if len(a) != len(b):
            print(f"DIFF {bf.name}: {len(a)} rows != {len(b)} rows"); bad += 1; continue
        for i, (x, y) in enumerate(zip(a, b)):
            d = first_diff(x, y, f"row{i}")
            if d:
                print(f"DIFF {bf.name}: {d}"); bad += 1; break
        else:
            print(f"OK   {bf.name} ({len(a)} rows)")
    for bf in sorted(base.glob("*.npz")):
        cf = cand / bf.name
        if not cf.exists():
            print(f"MISSING {bf.name}"); bad += 1; continue
        A, B = np.load(bf, allow_pickle=False), np.load(cf, allow_pickle=False)
        keys = sorted(k for k in set(A.files) | set(B.files) if not COLLEAGUE_KEY.search(k))
        diffs = [k for k in keys if k not in A.files or k not in B.files
                 or not np.array_equal(A[k], B[k], equal_nan=A[k].dtype.kind == "f")]
        if diffs:
            print(f"DIFF {bf.name}: arrays {diffs[:5]}"); bad += 1
        else:
            print(f"OK   {bf.name} ({len(keys)} arrays)")
    print("PASS" if bad == 0 else f"FAIL ({bad})")
    return 1 if bad else 0

if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
