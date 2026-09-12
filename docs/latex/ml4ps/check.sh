#!/bin/bash
# Pre-flight for the ML4PS manuscript. Run from anywhere after ANY edit to main.tex or the figure.
# 1. banned language  2. Times-metric build (XeLaTeX + Liberation Serif, metric-compatible with Times)
# 3. page-5 spill count in that build  4. LaTeX errors / unresolved refs  5. CM fallback preview (pdflatex).
set -u
cd "$(dirname "$0")"
fail=0
echo "== banned words (must be 0) =="
n=$(grep -c -E "colleague|\bhis\b|\bHis\b" main.tex); echo "main.tex: $n"; [ "$n" -eq 0 ] || fail=1
echo "== Times-metric build (xelatex, Liberation Serif; the official build uses Adobe Times, same widths within ~1%) =="
B=$(mktemp -d); cp main.tex references.bib neurips_2026.sty "$B"/; mkdir -p "$B/figures"; cp figures/*.pdf "$B/figures/"
python3 - "$B/main.tex" <<'PY'
import sys; p=sys.argv[1]; s=open(p).read()
old=r"\renewcommand{\rmdefault}{cmr}\renewcommand{\sfdefault}{cmss}\renewcommand{\ttdefault}{cmtt}"
assert s.count(old)==1
s=s.replace(old, r"\usepackage{fontspec}\setmainfont{Liberation Serif}\setsansfont{Liberation Sans}\setmonofont{Liberation Mono}")
s=s.replace(r"\usepackage[utf8]{inputenc}", "").replace(r"\usepackage[T1]{fontenc}", "")
open(p,"w").write(s)
PY
( cd "$B" && xelatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" >/dev/null 2>&1 \
  && bibtex main >/dev/null 2>&1 \
  && xelatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" >/dev/null 2>&1 \
  && xelatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" > build.log 2>&1 )
errs=$(grep -c "^!" "$B/build.log"); echo "LaTeX errors: $errs"; [ "$errs" -eq 0 ] || fail=1
grep -i "undefined" "$B/build.log" | head -3
pages=$(pdfinfo "$B/main.pdf" | awk '/Pages/{print $2}'); echo "pages: $pages"
spill=$(pdftotext -f 5 -l 5 "$B/main.pdf" - 2>/dev/null | awk '/^References/{exit} !/^[0-9 ]*$/{n++} END{print n+0}')
refpage=$(for pg in 4 5; do pdftotext -f $pg -l $pg "$B/main.pdf" - | grep -q '^References' && echo $pg; done | head -1)
echo "text lines on page 5 before References (Times-metric): $spill   [must be 0: References must start on page 4 or main text end on page 4]"
[ "$spill" -eq 0 ] || { echo "TOO LONG"; fail=1; }
cp "$B/main.pdf" preview_times_metric.pdf
echo "== CM fallback preview (pdflatex, wider font; informational) =="
C=$(mktemp -d); cp main.tex references.bib neurips_2026.sty "$C"/; mkdir -p "$C/figures"; cp figures/*.pdf "$C/figures/"
( cd "$C" && pdflatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" >/dev/null 2>&1 \
  && bibtex main >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" >/dev/null 2>&1 \
  && pdflatex -interaction=nonstopmode "\def\LOCALFONTCHECK{1}\input{main.tex}" >/dev/null 2>&1 )
cp "$C/main.pdf" preview_cm_fallback.pdf 2>/dev/null
echo "CM fallback page-5 text lines before References: $(pdftotext -f 5 -l 5 "$C/main.pdf" - 2>/dev/null | awk '/^References/{exit} !/^[0-9 ]*$/{n++} END{print n+0}')"
echo "== tables/figures reference the records? (manual) see COMPLIANCE.md =="
[ "$fail" -eq 0 ] && echo "CHECK PASS" || { echo "CHECK FAIL"; exit 1; }
