# --- removed from 09_physics_probe_facing_run.py:1-1 ---
# docstring title mentioned both instruments (verbatim original line)
"""Probe-facing curvature on the Physics anchors: <w_N, II> from both instruments against local R^2.


# --- removed from 09_physics_probe_facing_run.py:7-8 ---
# docstring PURPOSE paragraph: "Neither production instrument outputs that quantity. This
# runner computes it ... from both:" (fix round 1: was missing from the initial archive)
samplings where ``||H_tan||``'s partial swings from -0.29 to +0.26. Neither production instrument
outputs that quantity. This runner computes it on the real Physics anchors from both:


# --- removed from 09_physics_probe_facing_run.py:10-14 ---
# docstring: colleague instrument description in the two-item list
  decoder    : II = P_N D^2F at the anchor's latent code, by autodiff of the Amendment 01
               sphere-projected decoder (the frozen fit protocol, one fit per d);
  colleague  : his fitted sphere-normal quadratic B^S (his code, unchanged), averaged over the
               two halves and three splits, with his own `project_normal` for w_N and his own
               `probe_facing_scalar` reported beside our Frobenius contraction.


# --- removed from 09_physics_probe_facing_run.py:20-20 ---
# docstring: reference to the colleague's association
reported, since it removed the colleague's ||H_tan||-based association there.


# --- removed from 09_physics_probe_facing_run.py:24-24 ---
# docstring: colleague columns description
``bias_sq`` = (y - yhat_oof)^2 at the anchor, and for the colleague ``K_H_cross_col``


# --- removed from 09_physics_probe_facing_run.py:25-27 ---
# docstring COLUMNS description: colleague pf_curv_col/K_w_dir_col continuation
# (fix round 1: was missing from the initial archive)
(reference), ``pf_curv_col`` = |<w_N, B^S>|_F, ``K_w_dir_col`` = his probe-facing scalar times
|w_N|. Each column gets raw Spearman, the sealed partial, the multi-scale partial, and its
coupling with log radius and with ||H_tan||.


# --- removed from 09_physics_probe_facing_run.py:33-35 ---
# docstring usage examples used --skip-colleague / --colleague-root
    python notebooks/diagnostics/09_physics_probe_facing_run.py --mode smoke --skip-colleague --threads 8
    EFFDIM_09_OUTPUT_ROOT=... HF_HOME=... python notebooks/diagnostics/09_physics_probe_facing_run.py \\
        --mode physics --d-values 16,20 --colleague-root <root> --threads 16


# --- removed from 09_physics_probe_facing_run.py:48-49 ---
# module loader obtained `colleague` from `adj`
_spec.loader.exec_module(adj)          # loads the colleague runner, which loads the production runner (threads cap)
colleague, runner = adj.colleague, adj.runner


# --- removed from 09_physics_probe_facing_run.py:69-69 ---
# PRODUCTION_STEMS included the colleague runner's stem (original line, before trimming)
PRODUCTION_STEMS = ("09_physics_curvature", "09_colleague_estimator", "09_instrument_adjudication")


# --- removed from 09_physics_probe_facing_run.py:72-72 ---
# COL_COLUMNS constant (colleague-only column names)
COL_COLUMNS = ("K_H_cross_col", "pf_curv_col", "K_w_dir_col")


# --- removed from 09_physics_probe_facing_run.py:172-213 ---
# colleague_BS_at_anchors() and colleague_probe_facing()
# --- colleague: B^S at the anchors -----------------------------------------------------------


def colleague_BS_at_anchors(X: np.ndarray, neigh: np.ndarray, d: int, est: Dict[str, Any], device: torch.device,
                            n_splits: int, seed: int) -> Dict[str, Any]:
    """His `nested_pca_frame` + `_fit_rank` per anchor (unchanged), keeping the fitted B^S of both
    halves of every split and averaging them (his H_mean is the same average of the halves).
    Returns per-anchor x0 (b,D), J (b,D,d), BS_flat mean (b,D,q), K_H_cross (b), n_splits_ok."""
    nested_pca_frame, _fit_rank, _rows_from_fits = est["nested_pca_frame"], est["_fit_rank"], est["_rows_from_fits"]
    n_anchors, k = neigh.shape
    D = X.shape[1]; q = d * (d + 1) // 2
    x0s = np.zeros((n_anchors, D)); Js = np.zeros((n_anchors, D, d)); BS = np.full((n_anchors, D, q), np.nan)
    kh = np.full(n_anchors, np.nan); ok = np.zeros(n_anchors, dtype=int)
    t0 = time.monotonic()
    for ai in range(n_anchors):
        Xloc = X[neigh[ai, :k]].astype(np.float64)
        x0, J, _ev, _diag = nested_pca_frame(Xloc, d, device)
        fits = _fit_rank(Xloc, x0, J, d, k, n_splits, seed, ai)
        x0s[ai] = x0; Js[ai] = J[:, :d]
        if fits:
            rec = _rows_from_fits(ai, d, k, fits)
            kh[ai] = float(rec["K_H_cross"]); ok[ai] = int(rec.get("n_splits_ok", len(fits)))
            BS[ai] = np.mean([0.5 * (f["BS_flat_A"] + f["BS_flat_B"]) for f in fits], axis=0)
        if (ai + 1) % 64 == 0 or ai + 1 == n_anchors:
            print(f"[colleague] {ai + 1}/{n_anchors} anchors, {time.monotonic() - t0:.0f}s", flush=True)
    return {"x0": x0s, "J": Js, "BS_flat": BS, "K_H_cross": kh, "n_splits_ok": ok, "wallclock_s": time.monotonic() - t0}


def colleague_probe_facing(cb: Dict[str, Any], w: np.ndarray, d: int, est_mod: Dict[str, Any]) -> Dict[str, np.ndarray]:
    unpack, probe_facing_scalar, project_normal = est_mod["unpack_BS_symmetric"], est_mod["probe_facing_scalar"], est_mod["project_normal"]
    n = cb["BS_flat"].shape[0]
    pf = np.full(n, np.nan); kw = np.full(n, np.nan); wn = np.full(n, np.nan)
    for ai in range(n):
        if not np.all(np.isfinite(cb["BS_flat"][ai])):
            continue
        wn_unit, wn_norm = project_normal(w, cb["x0"][ai], cb["J"][ai])       # his: orthogonal to span(x0, J)
        B = unpack(cb["BS_flat"][ai], d)                                       # (D, d, d)
        b = np.einsum("a,aij->ij", wn_norm * wn_unit, B)
        pf[ai] = float(np.linalg.norm(b))                                      # his J is orthonormal: plain Frobenius
        kw[ai] = float(probe_facing_scalar(cb["BS_flat"][ai], d, wn_unit)["K_w_dir"]) * wn_norm
        wn[ai] = wn_norm
    return {"pf_curv_col": pf, "K_w_dir_col": kw, "w_N_norm_col": wn}


# --- removed from 09_physics_probe_facing_run.py:251-252 ---
# --colleague-root / --skip-colleague CLI arguments
    p.add_argument("--colleague-root", type=str, default=None)
    p.add_argument("--skip-colleague", action="store_true")


# --- removed from 09_physics_probe_facing_run.py:261-262 ---
# argument validation requiring --colleague-root unless --skip-colleague
    if not args.skip_colleague and args.colleague_root is None:
        raise SystemExit("--colleague-root is required unless --skip-colleague")


# --- removed from 09_physics_probe_facing_run.py:274-280 ---
# main(): loaded the colleague estimator and its geometry module
    est = est_mod = None
    if not args.skip_colleague:
        est = colleague.load_colleague_estimator(args.colleague_root)
        from geometry.physics_activation_atlas.confirmatory_object_curvature import unpack_BS_symmetric  # noqa: E402
        from geometry.physics_activation_atlas.effdim_curvature_metrics import probe_facing_scalar, project_normal  # noqa: E402
        est_mod = {"unpack_BS_symmetric": unpack_BS_symmetric, "probe_facing_scalar": probe_facing_scalar, "project_normal": project_normal}
        print(f"colleague checkout HEAD={est['colleague_head']} (expected {colleague.COLLEAGUE_COMMIT}); topology shim={est['topology_is_shim']}")


# --- removed from 09_physics_probe_facing_run.py:290-290 ---
# colleague_head field in the environment row (original line)
             "repo_head": adj._git_head(NOTEBOOK_ROOT.parent), "colleague_head": est["colleague_head"] if est else None,


# --- removed from 09_physics_probe_facing_run.py:321-323 ---
# neigh computed via colleague.colleague_neighbourhoods()
    neigh = None
    if est is not None:
        neigh = colleague.colleague_neighbourhoods(X, a, k)["neigh"]


# --- removed from 09_physics_probe_facing_run.py:343-347 ---
# cb = colleague_BS_at_anchors(...) per-d block
        cb = None
        if est is not None:
            cb = colleague_BS_at_anchors(X, neigh, d, est, torch.device(args.device), colleague.COLLEAGUE_N_SPLITS, colleague.COLLEAGUE_SEED)
            print(f"[colleague] K_H_cross finite {int(np.isfinite(cb['K_H_cross']).sum())}/{n_anchors}, {cb['wallclock_s']:.0f}s; "
                  f"rank(K_H_cross, H_tan_norm) = {_spearman(cb['K_H_cross'], dec['H_tan_norm']):+.3f}", flush=True)


# --- removed from 09_physics_probe_facing_run.py:353-354 ---
# colleague fields in the per-d 'fit' record append (original two lines)
                 "colleague_wallclock_s": cb["wallclock_s"] if cb else None,
                 "colleague_rank_vs_H_tan_norm": _spearman(cb["K_H_cross"], dec["H_tan_norm"]) if cb else None}, record_path)


# --- removed from 09_physics_probe_facing_run.py:361-363 ---
# colleague probe-facing columns merged into per-label cols
            if cb is not None:
                cpf = colleague_probe_facing(cb, L["w"], d, est_mod)
                cols["K_H_cross_col"] = cb["K_H_cross"]; cols.update({c: cpf[c] for c in ("pf_curv_col", "K_w_dir_col")})


