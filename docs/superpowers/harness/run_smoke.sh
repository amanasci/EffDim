#!/usr/bin/env bash
# Usage: run_smoke.sh <runners_dir> <out_dir> [colleague_root]
# Runs every kept runner in smoke mode, records to <out_dir>. Colleague flags passed only
# when the runner still accepts them.
set -uo pipefail
R=$1; OUT=$2; COL=${3:-}
PY=/home/akagi/Documents/Projects/EffDim/.venv/bin/python
mkdir -p "$OUT"; export EFFDIM_09_OUTPUT_ROOT="$OUT/output_root" PYTHONHASHSEED=0
col() { if [ -n "$COL" ] && grep -q -- '"--colleague-root"' "$R/$1.py"; then echo "--colleague-root $COL"; fi; }
run() { local name=$1; shift; local t0=$(date +%s)
  "$PY" "$R/$name.py" "$@" > "$OUT/$name.log" 2>&1; local rc=$?
  echo "$name rc=$rc $(( $(date +%s)-t0 ))s" | tee -a "$OUT/summary.txt"; }
: > "$OUT/summary.txt"
run 09_instrument_adjudication_run   --mode smoke $(col 09_instrument_adjudication_run) --threads 8 --record-path "$OUT/adj.jsonl"
run 09_fixture_probe_decodability_run --mode smoke $(col 09_fixture_probe_decodability_run) --threads 8 --record-path "$OUT/fix_dec.jsonl"
run 09_fixture_probe_facing_run      --mode smoke --threads 8 --record-path "$OUT/fix_facing.jsonl"
run 09_fixture_probe_facing_split_run --mode smoke --threads 8 --record-path "$OUT/fix_split.jsonl"
run 09_physics_probe_facing_run      --mode smoke $(col 09_physics_probe_facing_run) --threads 8 --record-path "$OUT/phys_facing.jsonl"
run 09_physics_probe_facing_split_run --mode smoke --threads 8 --record-path "$OUT/phys_split.jsonl"
run 09_physics_normal_scaling_run    --mode smoke --threads 8 --record-path "$OUT/ns.jsonl" --arrays-out "$OUT/ns_arrays.npz"
run 09_physics_probe_facing_split_run --mode smoke --threads 8 --hessian-xfit --record-path "$OUT/phys_split_xfit.jsonl"
run 09_physics_probe_facing_split_run --mode smoke --threads 8 --alpha 1 --record-path "$OUT/phys_split_alpha1.jsonl"
run 09_physics_normal_scaling_thin_run --help
