#!/usr/bin/env bash
# Usage: run_generators.sh <tree> <out_dir>
# Copies <tree> (without .git and caches) into a sandbox, links the real record cache into both
# notebooks/.cache and paper/.cache, runs every generator there, and collects: the spliced
# main.tex, table_main_gen stdout, and figure PNGs. Never writes inside <tree>.
set -euo pipefail
T=$(realpath "$1"); OUT=$2; PY=/home/akagi/Documents/Projects/EffDim/.venv/bin/python
REC=/home/akagi/Documents/Projects/EffDim/notebooks/.cache
W=${CLOSURE_WORK:-$HOME/.cache/effdim-closure}; mkdir -p "$W"; SB=$(mktemp -d "$W/sandbox.XXXX"); mkdir -p "$OUT"
rsync -a --exclude .git --exclude notebooks/.cache --exclude paper/.cache --exclude .venv \
  --exclude archive --exclude .planning --exclude docs/superpowers "$T/" "$SB/"
mkdir -p "$SB/notebooks" "$SB/paper"; ln -sfn "$REC" "$SB/notebooks/.cache"; ln -sfn "$REC" "$SB/paper/.cache"
export EFFDIM_CACHE_DIR="$REC" MPLBACKEND=Agg SOURCE_DATE_EPOCH=0
cd "$SB"
if [ -d paper/generate ]; then G=paper/generate; F=paper/latex/figures; TEX=paper/latex/main.tex
else G=docs/latex/ml4ps; F=docs/latex/ml4ps/figures; TEX=docs/latex/ml4ps/main.tex; fi
"$PY" $G/table_main_gen.py > "$OUT/table_main.txt"
"$PY" $G/appendix_gen.py > "$OUT/appendix_gen.log"
cp "$TEX" "$OUT/main.tex"
for s in make_fig1.py make_fig_intervention.py; do [ -f $F/$s ] && "$PY" $F/$s > "$OUT/$s.log"; done
cp $F/*.png "$OUT/" 2>/dev/null || true
rm -rf "$SB"; echo "generators done -> $OUT"
