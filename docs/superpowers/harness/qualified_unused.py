"""Module-qualified unused-symbol finder for pu_manifold (second pass after unused_symbols.py).

Usage: python qualified_unused.py --package <dir/pu_manifold> --root <file.py|file.ipynb> [...]

unused_symbols.py matches bare names, so a pu_manifold function survives whenever any kept
file uses the same name for something else (run_smoke, fit, build_arg_parser, ...). This pass
counts a use of module M's top-level NAME only when:
  - a kept file accesses it through an alias bound to M anywhere in the kept code
    (``from pu_manifold import M as A`` / ``import pu_manifold.M as A`` / ``from pu_manifold
    import M``), e.g. ``A.NAME`` or ``x.A.NAME``;
  - a kept file imports it directly (``from pu_manifold.M import NAME``);
  - a string constant equal to NAME appears in a kept file (getattr / monkeypatch.setattr);
  - a live definition in M itself mentions NAME (bare or attribute), iterated to a fixed point.
Module-level statements of M (non-definitions) are always live. Roots are the kept runners,
generators and notebooks passed with --root, plus every pu_manifold module (their cross-module
uses count through the same alias rules). Tests are never roots.

Prints, per module, the top-level symbols no rule keeps, with line ranges.
"""
import argparse, ast, json
from collections import defaultdict
from pathlib import Path


def source_of(p):
    p = Path(p)
    if p.suffix == ".ipynb":
        nb = json.loads(p.read_text())
        return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"
                         and not "".join(c["source"]).lstrip().startswith(("%", "!")))
    return p.read_text()


def defined(stmt):
    if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [stmt.name]
    if isinstance(stmt, ast.Assign):
        return [t.id for t in stmt.targets if isinstance(t, ast.Name)]
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return [stmt.target.id]
    return []


def is_main_guard(s):
    return (isinstance(s, ast.If) and isinstance(s.test, ast.Compare)
            and isinstance(s.test.left, ast.Name) and s.test.left.id == "__name__")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", required=True)
    ap.add_argument("--root", action="append", default=[])
    a = ap.parse_args()
    pkg = Path(a.package)
    modules = {p.stem: p for p in sorted(pkg.glob("*.py"))}
    trees = {f: ast.parse(source_of(f)) for f in a.root}
    mod_trees = {m: ast.parse(p.read_text()) for m, p in modules.items()}

    alias = defaultdict(set)          # alias identifier -> {module}
    direct = defaultdict(set)         # module -> names imported directly
    strings = set()
    for tree in list(trees.values()) + list(mod_trees.values()):
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == "pu_manifold":
                for x in n.names:
                    if x.name in modules: alias[x.asname or x.name].add(x.name)
            elif isinstance(n, ast.ImportFrom) and n.module and n.module.startswith("pu_manifold."):
                m = n.module.split(".", 1)[1]
                for x in n.names: direct[m].add(x.name)
            elif isinstance(n, ast.Import):
                for x in n.names:
                    if x.name.startswith("pu_manifold.") and x.asname:
                        alias[x.asname].add(x.name.split(".", 1)[1])
            elif isinstance(n, ast.Constant) and isinstance(n.value, str) and n.value.isidentifier():
                strings.add(n.value)

    def qualified_uses(node):
        out = defaultdict(set)
        for n in ast.walk(node):
            if isinstance(n, ast.Attribute):
                v = n.value
                key = v.id if isinstance(v, ast.Name) else v.attr if isinstance(v, ast.Attribute) else None
                for m in alias.get(key, ()):
                    out[m].add(n.attr)
        return out

    ext = defaultdict(set)
    for tree in trees.values():
        for m, names in qualified_uses(tree).items(): ext[m] |= names
    for m, names in direct.items(): ext[m] |= names

    cands = {}  # module -> list[(names, stmt)]
    static = defaultdict(set)
    for m, tree in mod_trees.items():
        cands[m] = []
        for s in tree.body:
            names = defined(s)
            if is_main_guard(s): continue
            if names and not any(k.startswith("__") for k in names): cands[m].append((names, s))
            else:
                static[m] |= {x.id for x in ast.walk(s) if isinstance(x, ast.Name)}
                for mm, nn in qualified_uses(s).items(): ext[mm] |= nn

    alive = {(m, id(s)) for m in cands for _, s in cands[m]}
    bare = {(m, id(s)): {x.id for x in ast.walk(s) if isinstance(x, ast.Name)} | {x.attr for x in ast.walk(s) if isinstance(x, ast.Attribute)}
            for m in cands for _, s in cands[m]}
    qual = {(m, id(s)): qualified_uses(s) for m in cands for _, s in cands[m]}
    changed = True
    while changed:
        changed = False
        used = defaultdict(set)
        for m in cands:
            used[m] |= ext[m] | static[m] | strings
        for m in cands:
            for _, s in cands[m]:
                k = (m, id(s))
                if k not in alive: continue
                for mm, nn in qual[k].items(): used[mm] |= nn
        for m in cands:
            for names, s in cands[m]:
                k = (m, id(s))
                if k not in alive: continue
                inner = set()
                for n2, s2 in cands[m]:
                    if s2 is not s and (m, id(s2)) in alive: inner |= bare[(m, id(s2))]
                if not any(n in used[m] or n in inner for n in names):
                    alive.discard(k); changed = True
    total = 0
    for m in cands:
        dead = [(names, s) for names, s in cands[m] if (m, id(s)) not in alive]
        lines = sum(s.end_lineno - s.lineno + 1 for _, s in dead); total += lines
        print(f"== {modules[m]}: {len(dead)} unused, {lines} lines")
        for names, s in dead:
            print(f"    {','.join(names)}  L{s.lineno}-{s.end_lineno}  ({s.end_lineno - s.lineno + 1} lines)")
    print(f"TOTAL unused lines: {total}")


if __name__ == "__main__":
    main()
