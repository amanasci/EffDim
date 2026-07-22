#!/usr/bin/env python3
"""CLI for SAE shared-basis experiments on cross-matched embedding pairs.

Examples
--------
  # list named datasets
  python experiments/SAE-shared-basis/run_shared_basis.py list

  # Ridge affine shared-basis (the result that worked well)
  python experiments/SAE-shared-basis/run_shared_basis.py run \\
      --dataset physics_vit_dino --experiment ridge

  # Custom pair (absolute or relative to --platonic-root)
  python experiments/SAE-shared-basis/run_shared_basis.py run \\
      --experiment ridge \\
      --parquet1 data_hf/physics/vit_base_test.parquet --col1 vit_base_galaxies \\
      --parquet2 data_hf/physics/dinov3_vitb16_test.parquet --col2 dinov3_vitb16_galaxies \\
      --sae-tag F2048_k64_seed0
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


HERE = Path(__file__).resolve().parent
DATASETS_YAML = HERE / "datasets.yaml"

EXPERIMENT_SCRIPTS = {
    "ridge": HERE / "sae_affine_basis_mknn_gpu.py",
    "lasso": HERE / "sae_affine_lasso_basis_mknn_gpu.py",
    "eigenbasis": HERE / "sae_lasso_eigenbasis_mknn_gpu.py",
}


def default_platonic_root() -> Path:
    env = os.environ.get("PLATONIC_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        Path("/home/angus/platonic-universe"),
        Path.home() / "platonic-universe",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def load_datasets() -> dict:
    if not DATASETS_YAML.is_file():
        raise FileNotFoundError(f"Missing {DATASETS_YAML}")
    if yaml is None:
        raise ImportError("PyYAML required: pip install pyyaml")
    data = yaml.safe_load(DATASETS_YAML.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError("datasets.yaml must be a mapping of name -> config")
    return data


def resolve(root: Path, p: str | Path) -> Path:
    path = Path(p).expanduser()
    return path if path.is_absolute() else (root / path)


def infer_sae_dir(root: Path, parquet: Path, col: str, sae_tag: str) -> Path:
    """outputs/sae/<parquet_stem>/<column>/<sae_tag>"""
    return root / "outputs" / "sae" / parquet.stem / col / sae_tag


def resolve_pair(
    root: Path,
    *,
    name: str | None,
    parquet1: str | None,
    col1: str | None,
    parquet2: str | None,
    col2: str | None,
    sae1: str | None,
    sae2: str | None,
    sae_tag: str | None,
) -> dict:
    cfg: dict = {}
    if name:
        catalog = load_datasets()
        if name not in catalog:
            known = ", ".join(sorted(catalog))
            raise SystemExit(f"Unknown dataset {name!r}. Known: {known}")
        cfg = dict(catalog[name])
        cfg["name"] = name

    # CLI overrides
    for key, val in [
        ("parquet1", parquet1),
        ("col1", col1),
        ("parquet2", parquet2),
        ("col2", col2),
        ("sae1", sae1),
        ("sae2", sae2),
    ]:
        if val is not None:
            cfg[key] = val
    if sae_tag is not None:
        cfg["sae_tag"] = sae_tag

    required = ["parquet1", "col1", "parquet2", "col2"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise SystemExit(
            f"Missing {missing}. Pass --dataset NAME or explicit --parquet*/--col*."
        )

    p1 = resolve(root, cfg["parquet1"])
    p2 = resolve(root, cfg["parquet2"])
    tag = cfg.get("sae_tag", "F2048_k64_seed0")
    s1 = resolve(root, cfg["sae1"]) if "sae1" in cfg else infer_sae_dir(root, p1, cfg["col1"], tag)
    s2 = resolve(root, cfg["sae2"]) if "sae2" in cfg else infer_sae_dir(root, p2, cfg["col2"], tag)

    return {
        "name": cfg.get("name", "custom"),
        "description": cfg.get("description", ""),
        "note": cfg.get("note", ""),
        "parquet1": p1,
        "col1": cfg["col1"],
        "parquet2": p2,
        "col2": cfg["col2"],
        "sae1": s1,
        "sae2": s2,
        "sae_tag": tag,
        "default_max_n": int(cfg.get("default_max_n", 16384)),
    }


def check_pair(pair: dict, *, strict_sae: bool) -> list[str]:
    errors = []
    for key in ("parquet1", "parquet2"):
        if not pair[key].is_file():
            errors.append(f"missing {key}: {pair[key]}")
    for key in ("sae1", "sae2"):
        d = pair[key]
        ok = (d / "model.pt").is_file() and (d / "config.json").is_file()
        if not ok:
            msg = f"missing SAE ({key}): {d}"
            if strict_sae:
                errors.append(msg)
            else:
                errors.append("WARN " + msg)
    return errors


def cmd_list(_args: argparse.Namespace) -> int:
    catalog = load_datasets()
    root = default_platonic_root()
    print(f"platonic-root: {root}")
    print(f"catalog: {DATASETS_YAML}\n")
    for name, cfg in sorted(catalog.items()):
        print(f"{name}")
        if cfg.get("description"):
            print(f"  {cfg['description']}")
        pair = resolve_pair(
            root,
            name=name,
            parquet1=None,
            col1=None,
            parquet2=None,
            col2=None,
            sae1=None,
            sae2=None,
            sae_tag=None,
        )
        print(f"  {pair['col1']}  ↔  {pair['col2']}")
        print(f"  parquet1: {pair['parquet1']}")
        print(f"  parquet2: {pair['parquet2']}")
        print(f"  sae1: {pair['sae1']}  [{'ok' if (pair['sae1']/'model.pt').is_file() else 'MISSING'}]")
        print(f"  sae2: {pair['sae2']}  [{'ok' if (pair['sae2']/'model.pt').is_file() else 'MISSING'}]")
        if cfg.get("note"):
            print(f"  note: {cfg['note']}")
        print()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    root = Path(args.platonic_root).expanduser().resolve()
    pair = resolve_pair(
        root,
        name=args.dataset,
        parquet1=args.parquet1,
        col1=args.col1,
        parquet2=args.parquet2,
        col2=args.col2,
        sae1=args.sae1,
        sae2=args.sae2,
        sae_tag=args.sae_tag,
    )
    print(f"dataset: {pair['name']}")
    if pair["description"]:
        print(pair["description"])
    for k in ("parquet1", "col1", "parquet2", "col2", "sae1", "sae2"):
        print(f"  {k}: {pair[k]}")
    errs = check_pair(pair, strict_sae=True)
    if errs:
        print("\nProblems:")
        for e in errs:
            print(f"  - {e}")
        return 1
    print("\nOK — paths look runnable.")
    return 0


def output_dir_for(pair: dict, experiment: str, max_n: int, out_root: Path) -> Path:
    tag = pair["sae_tag"].replace("/", "_")
    ntag = f"n{max_n}" if max_n and max_n > 0 else "nall"
    return out_root / f"{experiment}_{pair['name']}_{ntag}_{tag}"


def cmd_run(args: argparse.Namespace) -> int:
    root = Path(args.platonic_root).expanduser().resolve()
    pair = resolve_pair(
        root,
        name=args.dataset,
        parquet1=args.parquet1,
        col1=args.col1,
        parquet2=args.parquet2,
        col2=args.col2,
        sae1=args.sae1,
        sae2=args.sae2,
        sae_tag=args.sae_tag,
    )
    errs = check_pair(pair, strict_sae=not args.allow_missing_sae)
    hard = [e for e in errs if not e.startswith("WARN ")]
    soft = [e[5:] for e in errs if e.startswith("WARN ")]
    for w in soft:
        print(f"warning: {w}", file=sys.stderr)
    if hard:
        for e in hard:
            print(f"error: {e}", file=sys.stderr)
        print("hint: train SAEs or pass --sae1/--sae2; use doctor to inspect", file=sys.stderr)
        return 1

    script = EXPERIMENT_SCRIPTS[args.experiment]
    if not script.is_file():
        raise SystemExit(f"Missing experiment script: {script}")

    max_n = args.max_n
    if max_n is None:
        max_n = pair["default_max_n"]

    if args.output_dir:
        out_dir = Path(args.output_dir).expanduser()
        if not out_dir.is_absolute():
            out_dir = root / out_dir
    else:
        out_dir = output_dir_for(
            pair,
            args.experiment,
            int(max_n),
            root / "outputs" / "sae_shared_basis",
        )

    cmd = [
        sys.executable,
        str(script),
        "--parquet1",
        str(pair["parquet1"]),
        "--col1",
        pair["col1"],
        "--parquet2",
        str(pair["parquet2"]),
        "--col2",
        pair["col2"],
        "--sae1",
        str(pair["sae1"]),
        "--sae2",
        str(pair["sae2"]),
        "--max-n",
        str(int(max_n)),
        "--k",
        str(args.k),
        "--seed",
        str(args.seed),
        "--device",
        args.device,
        "--output-dir",
        str(out_dir),
    ]
    if args.experiment == "ridge":
        cmd += ["--alpha", str(args.alpha), "--test-size", str(args.test_size)]
    elif args.experiment == "lasso":
        cmd += ["--test-size", str(args.test_size)]
        if args.skip_mtl:
            cmd.append("--skip-mtl")
    elif args.experiment == "eigenbasis":
        cmd += ["--test-size", str(args.test_size), "--ridge-alpha", str(args.alpha)]
        if args.skip_c:
            cmd.append("--skip-C")

    if args.extra:
        cmd.extend(args.extra)

    print("Running:", " ".join(cmd), flush=True)
    print(f"output-dir: {out_dir}", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("PLATONIC_ROOT", str(root))
    # scripts hardcode platonic-universe for relative paths; we pass absolutes.
    return subprocess.call(cmd, env=env)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Run SAE shared-basis experiments on cross-matched datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--platonic-root",
        default=str(default_platonic_root()),
        help="Data/outputs root (env PLATONIC_ROOT overrides default search)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("list", help="List named datasets in datasets.yaml")
    sp.set_defaults(func=cmd_list)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", default=None, help="Name from datasets.yaml")
    common.add_argument("--parquet1")
    common.add_argument("--col1")
    common.add_argument("--parquet2")
    common.add_argument("--col2")
    common.add_argument("--sae1", help="SAE checkpoint dir for model 1")
    common.add_argument("--sae2", help="SAE checkpoint dir for model 2")
    common.add_argument(
        "--sae-tag",
        default=None,
        help="SAE run folder name, e.g. F2048_k64_seed0 (used if --sae1/--sae2 omitted)",
    )

    sd = sub.add_parser("doctor", parents=[common], help="Validate paths for a pair")
    sd.set_defaults(func=cmd_doctor)

    sr = sub.add_parser("run", parents=[common], help="Launch an experiment")
    sr.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_SCRIPTS),
        default="ridge",
        help="ridge = affine Ridge shared basis (recommended); "
        "lasso / eigenbasis = follow-ups",
    )
    sr.add_argument("--max-n", type=int, default=None, help="Subsample size (0 = all)")
    sr.add_argument("--test-size", type=float, default=0.3)
    sr.add_argument("--alpha", type=float, default=1.0, help="Ridge α")
    sr.add_argument("--k", type=int, default=10, help="mKNN k")
    sr.add_argument("--seed", type=int, default=0)
    sr.add_argument("--device", default="cuda")
    sr.add_argument("--output-dir", default=None)
    sr.add_argument(
        "--allow-missing-sae",
        action="store_true",
        help="Do not abort if SAE dirs are missing (script will still fail)",
    )
    sr.add_argument("--skip-mtl", action="store_true", help="lasso: skip multi-task path")
    sr.add_argument("--skip-c", action="store_true", help="eigenbasis: skip protocol C")
    sr.add_argument(
        "extra",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to the experiment script (use -- before them)",
    )
    sr.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = build_parser().parse_args(argv)
    # REMAINDER keeps a leading '--'; strip it for subprocess
    if getattr(args, "extra", None):
        if args.extra and args.extra[0] == "--":
            args.extra = args.extra[1:]
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
