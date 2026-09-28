# --- removed from 09_instrument_adjudication_run.py:5-12 ---
# docstring PURPOSE paragraph: colleague instrument description and "for BOTH" framing
# (fix round 1: was missing from the initial archive)
01, sphere-projected decoder, ``09_physics_curvature_run.fit_and_field_at_anchors``) and the
colleague's split-half nested-chart ``K_H_cross`` (``09_colleague_estimator_run``, his code
imported unchanged). Neither instrument has a known-answer validation in the regime where they
disagree: unit sphere in R^768, chart rank d=16, k=2048 neighbourhoods, n=86,471, noisy samples.
This runner supplies that known answer for BOTH, on the same points and the same anchors, and
scores each against it. It follows the spike-findings-effdim validation protocol: anchor at low d
first (``--mode swiss-roll``), state the pass regime in r/R, write the decision rule before the
numbers, score with the sealed four axes where they apply.


# --- removed from 09_instrument_adjudication_run.py:34-48 ---
# docstring WHAT IS AND IS NOT VALIDATED + SCORING paragraphs: two-estimator framing
# (fix round 1: was missing from the initial archive)
WHAT IS AND IS NOT VALIDATED. This validates the two estimators AS CURVATURE ESTIMATORS in the
Phase 9 regime: does each recover the ordering (and, for ours, the direction) of a known
sphere-intrinsic mean curvature field from n=86,471 samples at k=2048 and d=16, with and without
sample noise, on identical anchors? It does NOT validate or reinterpret the Physics result, the
Phase 9 verdict, or either pipeline's statistics; it touches no Phase 9 production record.

SCORING. Ours: ``synthetic_control_run._fidelity_axes`` (the sealed four axes: direction median
cosine, magnitude median ratio and CV, calibration slope/intercept/R^2, rank Spearman) of the
estimated ``H_tan`` vector against the truth ``H_tan`` vector, both projected to the tangent of
their own sphere image. His: rank Spearman of ``K_H_cross`` against ``||H_tan||^2 / d^2`` (his
``H`` is the diagonal MEAN of the second fundamental form, the averaged convention, and
``K_H_cross`` is the split-half inner product <H_A, H_B>, i.e. ||H_avg||^2 when the halves
agree; rank is invariant to that monotone map, so the rank comparison is convention-free) and a
scalar calibration against the same truth. Both: Spearman with ``log_knn_radius`` (density
coupling) beside the truth's own Spearman with log radius.


# --- removed from 09_instrument_adjudication_run.py:52-53 ---
# docstring DECISION RULES sphere-fixture line: "for ours also" / "per instrument"
# (fix round 1: was missing from the initial archive)
  sphere-fixture: rank rho vs truth >= 0.7, and for ours also direction median cosine >= 0.8
                 -> "validated in regime" per instrument per noise level.


# --- removed from 09_instrument_adjudication_run.py:59-62 ---
# docstring Usage examples used --colleague-root on every invocation
# (fix round 1: was missing from the initial archive)
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode smoke --colleague-root <root>
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode swiss-roll --colleague-root <root>
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode sphere-fixture --noise 0 --colleague-root <root> --threads 16
    python notebooks/diagnostics/09_instrument_adjudication_run.py --mode sphere-fixture --noise patch --colleague-root <root> --threads 16


# --- removed from 09_instrument_adjudication_run.py:69-81 ---
# module loader (Step 2 replaced this with a direct load of 09_physics_curvature_run)
DIAGNOSTICS_ROOT = Path(__file__).resolve().parent
NOTEBOOK_ROOT = DIAGNOSTICS_ROOT.parent
_COLLEAGUE_RUNNER_PATH = DIAGNOSTICS_ROOT / "09_colleague_estimator_run.py"

# Load the colleague runner first. It loads the production runner (09_physics_curvature_run)
# before numpy/torch are imported anywhere in this process, which applies the `--threads` cap
# from sys.argv (OMP/MKL/NUMEXPR env vars, then torch.set_num_threads) and puts notebooks/ and
# notebooks/diagnostics/ on sys.path. Same mechanism, called rather than copied.
_spec = importlib.util.spec_from_file_location("colleague_estimator_run", _COLLEAGUE_RUNNER_PATH)
colleague = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(colleague)
runner = colleague.runner



# --- removed from 09_instrument_adjudication_run.py:103-103 ---
# PRODUCTION_STEMS included the colleague runner's stem
PRODUCTION_STEMS = ("09_physics_curvature", "09_colleague_estimator")


# --- removed from 09_instrument_adjudication_run.py:297-313 ---
# his-instrument section header + colleague_field()
# --- his instrument -------------------------------------------------------------------------


def colleague_field(X: np.ndarray, anchor_idx: np.ndarray, k: int, d: int, est: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Exactly the colleague runner's path: self-excluded k rows from the sealed k-NN panel,
    nested PCA frame, split-half quadratic fits, n_splits=3, seed=0."""
    nb = colleague.colleague_neighbourhoods(X, anchor_idx, k)
    t0 = time.monotonic()
    curv = colleague.colleague_curvature_at_anchors(
        X, nb["neigh"], (d,), est, device, colleague.COLLEAGUE_N_SPLITS, colleague.COLLEAGUE_SEED,
    )
    return {
        "K_H_cross": curv[d]["K_H_cross"], "R_H": curv[d]["R_H"], "n_splits_ok": curv[d]["n_splits_ok"],
        "n_self_first": nb["n_self_first"], "wallclock_s": time.monotonic() - t0,
    }




# --- removed from 09_instrument_adjudication_run.py:324-333 ---
# score_his()
def score_his(kh: np.ndarray, truth_norm: np.ndarray, d: int, log_knn_radius: np.ndarray) -> Dict[str, Any]:
    truth = truth_norm ** 2 / d ** 2  # his averaged convention, squared; rank-invariant
    calib = _scalar_calibration(kh, truth)
    return {
        "truth_definition": "||H_tan||^2 / d^2 (averaged-convention squared norm)",
        "rank_spearman_rho": _spearman(kh, truth),
        "calibration_slope": calib["slope"], "calibration_intercept": calib["intercept"], "calibration_r2": calib["r2"],
        "rho_vs_log_knn_radius": _spearman(kh, log_knn_radius),
        "n_finite": int(np.isfinite(kh).sum()), "n_points": int(kh.shape[0]),
    }


# --- removed from 09_instrument_adjudication_run.py:354-355 ---
# colleague fields in _environment_row()'s returned dict
        "repo_head": _git_head(NOTEBOOK_ROOT.parent), "colleague_head": est["colleague_head"],
        "colleague_commit_expected": colleague.COLLEAGUE_COMMIT, "topology_is_shim": est["topology_is_shim"],


# --- removed from 09_instrument_adjudication_run.py:385-387 ---
# His: descriptive print in run_swiss_roll()
    print("His: nested_pca_frame/_fit_rank at d=2 on the self-excluded k rows. NOTE his frame normalises the "
          "neighbourhood mean to the unit sphere and projects the radial direction out of the tangent basis "
          "(a unit-sphere assumption); off-sphere it is being run outside its design regime, and this is reported as such.")


# --- removed from 09_instrument_adjudication_run.py:394-405 ---
# his computation + degenerate-check block in run_swiss_roll()
    his = colleague_field(X, anchor_idx, k, d, est, torch.device(args.device))
    his_scores = score_his(his["K_H_cross"], H_true_norm[anchor_idx], d, panel["log_knn_radius"])
    his_kh_absmax = float(np.nanmax(np.abs(his["K_H_cross"])))
    his_scores["K_H_cross_abs_max"] = his_kh_absmax
    his_scores["degenerate"] = bool(his_kh_absmax < 1e-12)
    print(f"[his] R_H median={float(np.nanmedian(his['R_H'])):.4f} finite={his_scores['n_finite']}/{n_anchors} "
          f"max|K_H_cross|={his_kh_absmax:.2e} {his['wallclock_s']:.1f}s ({his['wallclock_s'] / n_anchors:.2f}s/anchor)")
    if his_scores["degenerate"]:
        print("[his] DEGENERATE: K_H_cross is identically ~0. In R^3 at d=2 his frame projects out the unit-sphere "
              "normal x0/||x0|| AND a 2-dim tangent basis, leaving no normal direction for the quadratic fit to "
              "land in. The Swiss roll therefore cannot anchor his instrument; his on-sphere low-d anchor is the "
              "--mode smoke fixture (d=4 in S^63), whose verdict lines are informational.")


# --- removed from 09_instrument_adjudication_run.py:413-413 ---
# his_K_H_cross row entry in run_swiss_roll()'s printed table
        {"instrument": "his_K_H_cross", **{k_: his_scores[k_] for k_ in ("rank_spearman_rho", "calibration_slope", "calibration_r2", "rho_vs_log_knn_radius")}},


# --- removed from 09_instrument_adjudication_run.py:422-422 ---
# verdict tuple included a ("his", ...) entry in run_swiss_roll()
    for name, rho in (("ours", ours_axes["rank_spearman_rho"]), ("his", his_scores["rank_spearman_rho"])):


# --- removed from 09_instrument_adjudication_run.py:433-434 ---
# his_K_H_cross record append in run_swiss_roll()
    _append({**base, "instrument": "his_K_H_cross", "R_H_median": float(np.nanmedian(his["R_H"])), "scores": his_scores,
             "verdict": "PASS" if verdicts["his"] else "FAIL"}, record_path)


# --- removed from 09_instrument_adjudication_run.py:496-501 ---
# his computation block in run_fixture()
    print(f"[his] colleague path k={k} d={d} n_splits={colleague.COLLEAGUE_N_SPLITS} seed={colleague.COLLEAGUE_SEED} ...", flush=True)
    his = colleague_field(X, anchor_idx, k, d, est, torch.device(args.device))
    his_scores = score_his(his["K_H_cross"], truth["H_tan_norm"], d, log_r)
    r_h_median = float(np.nanmedian(his["R_H"]))
    print(f"[his] R_H median={r_h_median:.4f} finite={his_scores['n_finite']}/{n_anchors} {his['wallclock_s']:.1f}s "
          f"({his['wallclock_s'] / n_anchors:.2f}s/anchor)")


# --- removed from 09_instrument_adjudication_run.py:507-508 ---
# his_K_H_cross row entry in run_fixture()'s printed table
        {"instrument": "his_K_H_cross", **{k_: his_scores[k_] for k_ in (
            "rank_spearman_rho", "calibration_slope", "calibration_intercept", "calibration_r2", "rho_vs_log_knn_radius")}},


# --- removed from 09_instrument_adjudication_run.py:514-515 ---
# AE var_explained print included his R_H median in run_fixture()
    print(f"AE var_explained={ours['var_explained']:.4f}   his R_H median={r_h_median:.4f}   "
          f"truth rho(||H_tan||, log r)={truth_rho_radius:.4f}\n")


# --- removed from 09_instrument_adjudication_run.py:521-524 ---
# his_pass computation + print in run_fixture()
    his_pass = _ok(his_scores["rank_spearman_rho"], REGIME_RANK_RHO_PASS)
    print(f"ours: rank rho={_fmt(ours_scores['rank_spearman_rho'])} >= {REGIME_RANK_RHO_PASS} and direction cos="
          f"{_fmt(ours_scores['direction_median_cosine'])} >= {REGIME_DIRECTION_COS_PASS}: {'PASS' if ours_pass else 'FAIL'} [noise={noise}]")
    print(f"his:  rank rho={_fmt(his_scores['rank_spearman_rho'])} >= {REGIME_RANK_RHO_PASS}: {'PASS' if his_pass else 'FAIL'} [noise={noise}]")


# --- removed from 09_instrument_adjudication_run.py:534-536 ---
# his_K_H_cross record append and his keys in run_fixture()'s return dict
    _append({**base, "instrument": "his_K_H_cross", "R_H_median": r_h_median, "n_self_first": his["n_self_first"],
             "wallclock_s": his["wallclock_s"], "scores": his_scores, "verdict": "PASS" if his_pass else "FAIL"}, record_path)
    return {"ours": ours_scores, "his": his_scores, "ours_pass": ours_pass, "his_pass": his_pass,


# --- removed from 09_instrument_adjudication_run.py:541-541 ---
# run_smoke() banner mentioned both instruments
    print("\n" + "=" * 78 + "\nSMOKE: tiny fixture, both noise levels, both instruments. Verdict lines are informational;\n"


# --- removed from 09_instrument_adjudication_run.py:547-547 ---
# run_smoke() also checked res["his"] for finiteness
        finite = finite and np.isfinite(res["his"]["rank_spearman_rho"])


# --- removed from 09_instrument_adjudication_run.py:558-558 ---
# --colleague-root CLI argument
    p.add_argument("--colleague-root", type=str, required=True, help="read-only checkout at COLLEAGUE_COMMIT")


# --- removed from 09_instrument_adjudication_run.py:575-578 ---
# main(): loaded and verified the colleague estimator checkout
    est = colleague.load_colleague_estimator(args.colleague_root)
    print(f"colleague checkout HEAD={est['colleague_head']} (expected {colleague.COLLEAGUE_COMMIT}); topology shim={est['topology_is_shim']}")
    if est["colleague_head"] != colleague.COLLEAGUE_COMMIT:
        print("WARNING: colleague checkout is not at COLLEAGUE_COMMIT")


