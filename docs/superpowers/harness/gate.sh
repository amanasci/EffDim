#!/usr/bin/env bash
# Usage: gate.sh <tree> <label>
# Stage gate: smoke runs + pytest + generators on <tree>, compared with $W/baseline
# (W = $CLOSURE_WORK, default ~/.cache/effdim-closure; holds baseline/, runs/, colleague/).
# Works on both layouts (notebooks/diagnostics or paper/runners). Exit 0 only if all pass.
set -uo pipefail
H=$(dirname "$(realpath "$0")"); W=${CLOSURE_WORK:-$HOME/.cache/effdim-closure}
T=$(realpath "$1"); L=$2; OUT=$W/runs/$L
PY=/home/akagi/Documents/Projects/EffDim/.venv/bin/python; COL=$W/colleague
rm -rf "$OUT"; mkdir -p "$OUT"; fail=0
if [ -d "$T/paper/runners" ]; then R=$T/paper/runners; TD=$T/paper; TP=tests
else R=$T/notebooks/diagnostics; TD=$T/notebooks; TP=pu_manifold/tests; fi
echo "== smoke ($R)"; "$H/run_smoke.sh" "$R" "$OUT/smoke" "$COL" >/dev/null
if grep -v " rc=0 " "$OUT/smoke/summary.txt"; then echo "SMOKE RUNNER FAILED (logs in $OUT/smoke)"; fail=1; fi
"$PY" "$H/compare.py" "$W/baseline/smoke" "$OUT/smoke" | tail -12 || fail=1
"$PY" "$H/compare.py" "$W/baseline/smoke" "$OUT/smoke" >/dev/null || fail=1
echo "== pytest ($TD/$TP)"; ( cd "$TD" && "$PY" -m pytest -q -p no:cacheprovider $TP ) > "$OUT/pytest.log" 2>&1 || fail=1
tail -1 "$OUT/pytest.log"
echo "== generators"; "$H/run_generators.sh" "$T" "$OUT/gen" >/dev/null || fail=1
"$PY" "$H/compare_generators.py" "$W/baseline/gen" "$OUT/gen" || fail=1
[ $fail -eq 0 ] && echo "GATE PASS ($L)" || echo "GATE FAIL ($L)"; exit $fail
