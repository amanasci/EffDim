"""Turn file:line references in a Markdown file into relative links with GitHub line anchors.

    python docs/audit/link_refs.py docs/audit/2026-10-05-code-logic-audit.md

A reference is a repo file name or path, optionally with :N, :N-M or :N,M lists (`appendix_gen.py:62,224`),
optionally in backticks. Bare names resolve against `git ls-files` when the match is unique (main.tex prefers
paper/latex/). A bare ":N" or ":N-M" later in the same paragraph links to the last file named there.
Unresolvable or git-ignored files (e.g. notebooks/.cache records) are left as text. Idempotent: text already
inside a Markdown link is skipped.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
TRACKED = subprocess.check_output(["git", "ls-files"], cwd=ROOT, text=True).split()
PREFER = ("paper/latex/", "curvature-experiment/", "docs/audit/2026-10-05-scripts/", "src/")
EXT = r"(?:py|tex|md|sh|yaml|jsonl|json|bib|txt|npz|sty)"
LINES = r"(?::\d+(?:-\d+)?(?:,\d+(?:-\d+)?)*)"
REF = re.compile(rf"(`?)(?<![\w/.*-])([A-Za-z0-9_][A-Za-z0-9_./-]*\.{EXT})({LINES})?\1(?![\w*])")
BARE = re.compile(r"(?<=[\s(`])(`?)(:\d+(?:-\d+)?)\1(?=[\s),.;`])")
LINK = re.compile(r"\[[^\]]*\]\([^)]*\)")


def resolve(name: str):
    if name in TRACKED:
        return name
    hits = [p for p in TRACKED if p == name or p.endswith("/" + name)]
    if len(hits) > 1:
        for pre in PREFER:
            pref = [p for p in hits if p.startswith(pre)]
            if len(pref) == 1:
                return pref[0]
    return hits[0] if len(hits) == 1 else None


def anchor(span: str) -> str:
    a, _, b = span.partition("-")
    return f"#L{a}-L{b}" if b else f"#L{a}"


def link_line(line: str, rel_base: Path, state: dict) -> str:
    """One left-to-right pass: file references set the current file; a bare :N links to the nearest earlier one."""
    protected = [(m.start(), m.end()) for m in LINK.finditer(line)]
    hits = [(m.start(), "ref", m) for m in REF.finditer(line)] + [(m.start(), "bare", m) for m in BARE.finditer(line)]
    hits.sort(key=lambda h: h[0])
    out, pos = [], 0
    for start, kind, m in hits:
        if start < pos or any(s <= start < e for s, e in protected):
            continue
        if kind == "ref":
            tick, name, lines = m.group(1), m.group(2), m.group(3) or ""
            path = resolve(name)
            if path is None:
                continue
            state["last"] = path
            href = os.path.relpath(ROOT / path, rel_base)
            parts = lines[1:].split(",") if lines else []
            first = f"{name}:{parts[0]}" if parts else name
            pieces = [f"[{tick}{first}{tick}]({href}{anchor(parts[0]) if parts else ''})"]
            pieces += [f"[{tick}{p}{tick}]({href}{anchor(p)})" for p in parts[1:]]
            text = ", ".join(pieces)
        else:
            if not state.get("last"):
                continue
            tick, span = m.group(1), m.group(2)
            href = os.path.relpath(ROOT / state["last"], rel_base)
            text = f"[{tick}{span}{tick}]({href}{anchor(span[1:])})"
        out.append(line[pos:m.start()]); out.append(text); pos = m.end()
    out.append(line[pos:])
    return "".join(out)


def main(md: str) -> None:
    p = Path(md).resolve()
    state, fence, new = {}, False, []
    for line in p.read_text().splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fence = not fence
        if not line.strip() or line.lstrip().startswith("|"):
            state = {}                      # bare ":N" refs attach within a prose paragraph or one table row
        new.append(line if fence else link_line(line, p.parent, state))
    p.write_text("".join(new))


if __name__ == "__main__":
    for f in sys.argv[1:]:
        main(f)
