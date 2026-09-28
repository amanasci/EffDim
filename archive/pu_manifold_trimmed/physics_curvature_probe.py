# Top-level definitions removed from notebooks/pu_manifold/physics_curvature_probe.py in the paper-closure
# Stage 5: not reachable from any kept runner, generator or notebook. Verbatim, original order.


# --- removed from physics_curvature_probe.py:924-949 ---
def stratified_partial_null_3control(
    x: np.ndarray, y: np.ndarray, Z: np.ndarray, strata_field: np.ndarray, n_strata: int, n_draws: int, seed: int
) -> Dict[str, Any]:
    """Bins with ``density_stratified_null.density_strata(strata_field, n_strata)``, then
    permutes ``x`` and ``y`` INDEPENDENTLY within each stratum per draw and calls
    :func:`controlled_partial` with the full 3-column ``Z`` inside the loop. Does not edit
    ``density_stratified_null.py`` to generalise its single-control ``stratified_partial_null``
    -- additive only."""
    strata = density_stratified_null.density_strata(strata_field, n_strata)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    observed = controlled_partial(x, y, Z)

    rng = np.random.default_rng(seed)
    null_draws = np.empty(n_draws, dtype=np.float64)
    for b in range(n_draws):
        xp = x.copy()
        yp = y.copy()
        for s in np.unique(strata):
            idx = np.where(strata == s)[0]
            xp[idx] = x[rng.permutation(idx)]
            yp[idx] = y[rng.permutation(idx)]
        null_draws[b] = controlled_partial(xp, yp, Z)

    pv = p_value_from_null(observed, null_draws)
    return {"observed": observed, "null_draws": null_draws, **pv}


# --- removed from physics_curvature_probe.py:952-966 ---
def paired_anchor_bootstrap(x: np.ndarray, y: np.ndarray, Z: np.ndarray, n_boot: int, seed: int) -> Dict[str, Any]:
    """Resamples anchor ROWS with replacement, carrying ``x``, ``y`` and every control column
    together so the pairing is preserved, recomputes :func:`controlled_partial` per draw, and
    returns the 2.5/97.5 percentile band and the draw count."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    Z = np.asarray(Z, dtype=np.float64)
    n = x.shape[0]
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        draws[b] = controlled_partial(x[idx], y[idx], Z[idx])
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"ci_low": float(lo), "ci_high": float(hi), "n_boot": int(n_boot), "draws": draws}


# --- removed from physics_curvature_probe.py:969-1044 ---
def plant_curvature_positive_control(
    h_real: np.ndarray, y: np.ndarray, Z: np.ndarray, target_rho: float, seed: int, n_bisect: int
) -> Dict[str, Any]:
    """Reuses ``crossmodal_curvature.plant_positive_control``'s MECHANISM (guard first: raise
    ``ValueError`` naming ``h_real`` before any search when it is constant or non-finite;
    rank-transform; bisect a slope over ``n_bisect`` iterations on the bracket ``[0.0, 2.0]``)
    but retargets the achieved statistic at
    ``controlled_partial(planted, y, Z)``, spread-matched to the realized range of ``h_real``.
    Returns the planted array, the achieved controlled partial, and the slope. The null
    validation is the caller's job and must be :func:`permutation_fwer`'s Freedman-Lane
    construction -- ``crossmodal_curvature.two_tailed_permutation_null`` is the wrong null for
    this phase.

    Direction note (a real adaptation, not present in the sealed mechanism): the sealed
    ``plant_positive_control`` always bisects assuming ``spearmanr(h_real, planted)`` INCREASES
    with slope, which holds unconditionally there because the achieved statistic is measured
    against ``h_real`` itself. Here the achieved statistic is measured against ``y`` (e.g. the
    local out-of-fold R2), so whether ``controlled_partial(planted, y, Z)`` increases or
    decreases with slope depends on the empirical sign of the ``h_real``-``y`` relationship --
    for this phase's own negative-association hypothesis (D9-09), it decreases. The direction is
    therefore measured once (achieved at slope 0.0 vs slope 2.0) before bisecting, rather than
    assumed fixed."""
    from scipy.stats import rankdata

    h = np.asarray(h_real, dtype=np.float64).ravel()
    if not np.all(np.isfinite(h)):
        raise ValueError("plant_curvature_positive_control: h_real contains a non-finite value.")
    if np.ptp(h) == 0:
        raise ValueError("plant_curvature_positive_control: h_real is constant (np.ptp(h_real) == 0).")

    n = h.shape[0]
    u = (rankdata(h) - 0.5) / n
    lo_val, hi_val = float(np.min(h)), float(np.max(h))
    spread = hi_val - lo_val
    # A small discretization (mirroring the sealed mechanism's own k-sized binomial trial count,
    # rather than an arbitrary fine-grained one) keeps controlled_partial(planted, y, Z) a
    # smooth, near-monotonic function of slope across the whole [0.0, 2.0] bracket; a much finer
    # discretization saturates the achieved statistic within the first few percent of the
    # bracket, making bisection unable to resolve intermediate targets.
    _discretization = 10

    def _planted(slope: float) -> np.ndarray:
        p = np.clip(0.5 + slope * (u - 0.5), 0.0, 1.0)
        rng_ = np.random.default_rng(seed)
        j = rng_.binomial(_discretization, p)
        return lo_val + spread * (j / float(_discretization))

    achieved_at_low = controlled_partial(_planted(0.0), y, Z)
    achieved_at_high = controlled_partial(_planted(2.0), y, Z)
    increasing = achieved_at_high >= achieved_at_low

    low, high = 0.0, 2.0
    for _ in range(n_bisect):
        mid = (low + high) / 2.0
        mid_planted = _planted(mid)
        mid_achieved = controlled_partial(mid_planted, y, Z)
        if increasing:
            if mid_achieved < target_rho:
                low = mid
            else:
                high = mid
        else:
            if mid_achieved > target_rho:
                low = mid
            else:
                high = mid

    slope = high
    planted = _planted(slope)
    achieved = controlled_partial(planted, y, Z)
    return {
        "planted": planted,
        "achieved_controlled_partial": float(achieved),
        "slope": float(slope),
        "target_rho": float(target_rho),
    }


# --- removed from physics_curvature_probe.py:1047-1084 ---
def shuffled_label_repeat(
    X: np.ndarray,
    y: np.ndarray,
    neighbour_idx: np.ndarray,
    log_knn_radius: np.ndarray,
    h_field: np.ndarray,
    alpha: float,
    n_folds: int,
    fold_seed: int,
    min_finite: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Permutes ``y`` across rows with ``rng``, recomputes the OOF predictions, recomputes
    :func:`local_r2_panel` and therefore BOTH label-derived controls, reuses the caller's
    ``log_knn_radius`` and ``h_field`` unchanged, and returns the controlled partial plus the
    masked count. The embedding matrix, the curvature field and the anchor index array are held
    byte-identical across repeats -- only the label vector moves."""
    y = np.asarray(y, dtype=np.float64).ravel()
    n = y.shape[0]
    perm = rng.permutation(n)
    y_shuffled = y[perm]

    y_hat = oof_ridge_predictions(X, y_shuffled, alpha, n_folds, fold_seed)
    panel = local_r2_panel(y_shuffled, y_hat, neighbour_idx, min_finite)

    controls = np.column_stack(
        [log_knn_radius, panel["local_label_variance"], panel["local_evaluation_count"]]
    )
    h_field = np.asarray(h_field, dtype=np.float64)
    finite = np.isfinite(panel["r2"])
    controlled = controlled_partial(h_field[finite], panel["r2"][finite], controls[finite])

    return {
        "controlled_partial": float(controlled),
        "local_label_variance": panel["local_label_variance"],
        "local_evaluation_count": panel["local_evaluation_count"],
        "n_masked_anchors": panel["n_masked_anchors"],
    }


# --- removed from physics_curvature_probe.py:1116-1128 ---
def combine_seed_verdicts(seed_verdicts: Any) -> str:
    """Raises ``ValueError`` unless given exactly three entries; returns the shared value on
    unanimity; returns ``"SPLIT ACROSS SEEDS"`` otherwise. Never averages, never upgrades a
    2-of-3. Mirrors ``05-03-DECISION.md``'s one-way ratification."""
    verdicts = list(seed_verdicts)
    if len(verdicts) != 3:
        raise ValueError(
            f"combine_seed_verdicts: expected exactly three seed verdict entries; got "
            f"{len(verdicts)}."
        )
    if len(set(verdicts)) == 1:
        return verdicts[0]
    return "SPLIT ACROSS SEEDS"


# --- removed from physics_curvature_probe.py:1131-1154 ---
def verdict_sentence(
    instrument: str,
    d_values: Any,
    colleague_rho: float,
    colleague_d: int,
    fwer_p_display: str,
    stratified_p_display: str,
    instrument_fidelity_ranges: Dict[int, Any],
    neighbourhood_ratio: str,
) -> str:
    """Assembles D9-10's caveat-bearing sentence naming the instrument, the ``d`` values, the
    colleague's ``-0.240`` at his ``d=16``, both nulls, the instrument-fidelity ranges, and the
    neighbourhood n-ratio. Reads :data:`VERDICT_SENTENCE_RULE`."""
    if _is_unset(VERDICT_SENTENCE_RULE):
        raise RuntimeError(
            "verdict_sentence: VERDICT_SENTENCE_RULE is UNSET; the freeze (09-05) must fill it "
            "before a verdict sentence can be assembled."
        )
    return (
        f"Instrument {instrument} at d={list(d_values)}, against the colleague's "
        f"{colleague_rho:.3f} at his d={colleague_d}: Freedman-Lane FWER p={fwer_p_display}, "
        f"density-stratified null p={stratified_p_display}; instrument fidelity ranges "
        f"{instrument_fidelity_ranges}; neighbourhood ratio {neighbourhood_ratio}."
    )
