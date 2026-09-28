# --- removed from 09_fixture_probe_decodability_run.py:23-23 ---
# docstring: colleague instrument description (verbatim original line)
     the colleague's split-half quadratic ``K_H^cross`` (his code, unchanged). Pointwise-vs-patch


# --- removed from 09_fixture_probe_decodability_run.py:41-44 ---
# docstring usage examples used --skip-colleague / --colleague-root
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode smoke --skip-colleague
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode full --gammas -1,0,1 \\
        --colleague-root <root> --threads 16
    python notebooks/diagnostics/09_fixture_probe_decodability_run.py --mode full --skip-decoder --skip-colleague


# --- removed from 09_fixture_probe_decodability_run.py:55-55 ---
# module loader comment mentioned the colleague runner
# The adjudication runner loads the colleague runner, which loads the production runner, which


# --- removed from 09_fixture_probe_decodability_run.py:60-60 ---
# module loader obtained `colleague` from `adj` (original line)
colleague, runner = adj.colleague, adj.runner


# --- removed from 09_fixture_probe_decodability_run.py:77-77 ---
# PRODUCTION_STEMS included the colleague runner's stem (original line, before trimming)
PRODUCTION_STEMS = ("09_physics_curvature", "09_colleague_estimator", "09_instrument_adjudication")


# --- removed from 09_fixture_probe_decodability_run.py:84-84 ---
# COLUMN_NAMES included colleague_K_H_cross (original line, before trimming)
COLUMN_NAMES = ("exact_point", "exact_patch", "decoder_H_tan", "colleague_K_H_cross")


# --- removed from 09_fixture_probe_decodability_run.py:212-212 ---
# run_gamma()'s est parameter (original signature line, before trimming)
              est: Optional[Dict[str, Any]], record_path: Path, max_epochs: int, n_perm: int) -> None:


# --- removed from 09_fixture_probe_decodability_run.py:253-260 ---
# colleague computation block inside run_gamma()
    if not args.skip_colleague:
        print(f"[colleague] k={k} d={d} n_splits={colleague.COLLEAGUE_N_SPLITS} ...", flush=True)
        his = adj.colleague_field(X, a, k, d, est, torch.device(args.device))
        columns["colleague_K_H_cross"] = his["K_H_cross"]
        fit_info["colleague"] = {"R_H_median": float(np.nanmedian(his["R_H"])), "wallclock_s": float(his["wallclock_s"]),
                                 "rank_vs_truth": _spearman(his["K_H_cross"], truth["H_tan_norm"] ** 2 / d ** 2)}
        print(f"[colleague] R_H median={fit_info['colleague']['R_H_median']:.3f} {his['wallclock_s']:.0f}s; "
              f"rank vs truth {fit_info['colleague']['rank_vs_truth']:.3f}", flush=True)


# --- removed from 09_fixture_probe_decodability_run.py:303-303 ---
# --colleague-root CLI argument (fix round 1: was missing from the initial archive)
    p.add_argument("--colleague-root", type=str, default=None, help="read-only checkout at COLLEAGUE_COMMIT")


# --- removed from 09_fixture_probe_decodability_run.py:305-305 ---
# --skip-colleague CLI argument (fix round 1: was missing from the initial archive)
    p.add_argument("--skip-colleague", action="store_true")


# --- removed from 09_fixture_probe_decodability_run.py:318-319 ---
# argument validation requiring --colleague-root unless --skip-colleague (fix round 1: was
# missing from the initial archive)
    if not args.skip_colleague and args.colleague_root is None:
        raise SystemExit("--colleague-root is required unless --skip-colleague")


# --- removed from 09_fixture_probe_decodability_run.py:328-331 ---
# main(): loaded and printed the colleague estimator checkout
    est = None
    if not args.skip_colleague:
        est = colleague.load_colleague_estimator(args.colleague_root)
        print(f"colleague checkout HEAD={est['colleague_head']} (expected {colleague.COLLEAGUE_COMMIT}); topology shim={est['topology_is_shim']}")


# --- removed from 09_fixture_probe_decodability_run.py:333-333 ---
# colleague_head field in the environment row (original line)
             "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "colleague_head": est["colleague_head"] if est else None,


