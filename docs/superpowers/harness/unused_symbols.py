"""Conservative, name-based unused-symbol finder for the paper closure.

Usage: python unused_symbols.py --entry <runner.py> [--entry ...] --library <file.py> [...]
                                [--notebook <nb.ipynb> ...] [--generator <gen.py> ...]

Generators are roots in full. Entry runners (invoked in REPRODUCE.md) root their main(), their
__main__ block and their module-level statements; their other top-level defs are candidates. Library files (pu_manifold modules, import-only runners) contribute
their module-level statements, but their top-level defs/classes/assignments are candidates.
A candidate NAME is used when NAME occurs as an identifier (ast.Name id, ast.Attribute attr,
keyword arg, or a string constant equal to NAME, which catches getattr/monkeypatch/__all__)
anywhere in scanned code other than inside NAME's own definition. Removal is iterated to a
fixed point: an unused def's body stops counting as a use. Library __main__ blocks are not
scanned. Name collisions across modules make the result conservative (keeps more), never
aggressive. Tests are deliberately not roots.

Prints, per library file, the unused top-level symbols with line ranges.
"""
import argparse, ast, json, sys
from pathlib import Path

def tokens(node):
    out = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name): out.add(n.id)
        elif isinstance(n, ast.Attribute): out.add(n.attr)
        elif isinstance(n, ast.keyword) and n.arg: out.add(n.arg)
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier(): out.add(n.value)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names: out.add(a.name.split(".")[-1]); out.add(a.asname or "")
    return out

def is_main_guard(stmt):
    return (isinstance(stmt, ast.If) and isinstance(stmt.test, ast.Compare)
            and isinstance(stmt.test.left, ast.Name) and stmt.test.left.id == "__name__")

def defined_names(stmt):
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)): return [stmt.name]
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name): return [stmt.target.id]
    return []

def notebook_source(p):
    nb = json.loads(Path(p).read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
                     and not "".join(c["source"]).lstrip().startswith(("%", "!")))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--entry", action="append", default=[])
    ap.add_argument("--generator", action="append", default=[])
    ap.add_argument("--notebook", action="append", default=[])
    ap.add_argument("--library", action="append", default=[])
    ap.add_argument("--ignore-name", action="append", default=[],
                    help="names that never count as a use (e.g. main, which every entry runner calls on itself)")
    a = ap.parse_args()
    root_tokens = set()
    for f in a.generator:
        root_tokens |= tokens(ast.parse(Path(f).read_text()))
    for f in a.notebook:
        root_tokens |= tokens(ast.parse(notebook_source(f)))
    cands = []  # (file, name, stmt)
    lib_static = set()
    forced = set()  # ids of entry-runner main() defs: always alive
    for f in a.entry + a.library:
        is_entry = f in a.entry
        tree = ast.parse(Path(f).read_text())
        for stmt in tree.body:
            names = defined_names(stmt)
            if is_main_guard(stmt):
                if is_entry: lib_static |= tokens(stmt)
                continue
            if is_entry and names == ["main"]:
                lib_static |= tokens(stmt); continue
            if names and not any(n.startswith("__") for n in names):
                cands.append((f, names, stmt))
            else:
                lib_static |= tokens(stmt)
    from collections import Counter
    toks = {id(s): tokens(s) for _, _, s in cands}
    alive = {id(s) for _, _, s in cands}
    base = (root_tokens | lib_static) - set(a.ignore_name)
    for k in list(toks): toks[k] = toks[k] - set(a.ignore_name)
    changed = True
    while changed:
        changed = False
        # how many live candidates mention each name; a def's own tokens are subtracted when
        # judging that def, so it never keeps itself alive
        mentions = Counter()
        for _, _, s in cands:
            if id(s) in alive: mentions.update(toks[id(s)])
        for f, names, s in cands:
            if id(s) not in alive: continue
            own = toks[id(s)]
            if not any(n in base or mentions[n] - (1 if n in own else 0) > 0 for n in names):
                alive.discard(id(s)); changed = True
    by_file = {}
    for f, names, s in cands:
        if id(s) not in alive:
            by_file.setdefault(f, []).append(f"{','.join(names)}  L{s.lineno}-{s.end_lineno}  ({s.end_lineno - s.lineno + 1} lines)")
    total = 0
    for f in a.entry + a.library:
        items = by_file.get(f, [])
        n = sum(int(x.split("(")[-1].split()[0]) for x in items); total += n
        print(f"== {f}: {len(items)} unused, {n} lines")
        for x in items: print("   ", x)
    print(f"TOTAL unused lines: {total}")

if __name__ == "__main__":
    main()
