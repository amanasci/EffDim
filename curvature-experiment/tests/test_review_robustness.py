"""Review robustness: concerns 2-5 on stored geometry."""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

RUNNERS = Path(__file__).resolve().parents[1] / "runners"


def _load(name):
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), RUNNERS / name)
    mod = importlib.util.module_from_spec(spec)
    argv, sys.argv = sys.argv, [name]
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.argv = argv
    return mod


rr = _load("11_review_robustness_run.py")


def _ridge_data(n=600, D=12, noise=0.5, seed=0, nan_frac=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, D)); beta = rng.standard_normal(D)
    y = X @ beta + noise * rng.standard_normal(n)
    if nan_frac:
        y[rng.random(n) < nan_frac] = np.nan
    return X, y


def test_oof_fixed_alpha_matches_published_helper():
    X, y = _ridge_data(nan_frac=0.1)
    got, alphas = rr.oof_predictions(X, y, (100.0, 100.0))
    want = rr.runner._oof_predictions_for_label(X, y, 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    assert np.array_equal(np.isnan(got), np.isnan(want))
    np.testing.assert_array_equal(got[np.isfinite(got)], want[np.isfinite(want)])
    assert alphas == [100.0] * rr.pcp.N_OOF_FOLDS


def test_tuned_oof_never_uses_held_out_rows():
    """Scrambling the labels of fold 0's held-out rows must not change the alpha chosen for fold 0."""
    from sklearn.model_selection import KFold
    X, y = _ridge_data()
    _, a_ref = rr.oof_predictions(X, y, rr.ALPHA_GRID)
    test0 = next(iter(KFold(rr.pcp.N_OOF_FOLDS, shuffle=True, random_state=rr.pcp.OOF_FOLD_SEED).split(X)))[1]
    y2 = y.copy(); y2[test0] = np.random.default_rng(9).standard_normal(len(test0)) * 100
    _, a_new = rr.oof_predictions(X, y2, rr.ALPHA_GRID)
    assert a_new[0] == a_ref[0]


def test_select_alpha_small_for_strong_signal():
    X, y = _ridge_data(noise=0.01)
    assert rr.select_alpha(X, y) <= 1.0
    assert rr.PUBLISHED_ALPHA in rr.ALPHA_GRID and 1.0 in [round(a, 12) for a in rr.ALPHA_GRID]


def test_global_probe_matches_ridge():
    from sklearn.linear_model import Ridge
    X, y = _ridge_data(nan_frac=0.1)
    w, b0 = rr.global_probe(X, y, 100.0)
    fin = np.isfinite(y)
    ref = Ridge(alpha=100.0).fit(X[fin], y[fin])
    np.testing.assert_array_equal(w, ref.coef_); assert b0 == float(ref.intercept_)


@pytest.fixture(scope="module")
def smoke():
    """The split runner's smoke fixture: generator data, a 3-epoch decoder, its geometry at the anchors."""
    data = rr.ppf.load_smoke(argparse.Namespace(seed=20260905))
    X = data["X"]; n = X.shape[0]
    k, n_anchors, d = rr.adj.SMOKE["k"], rr.adj.SMOKE["n_anchors"], rr.adj.SMOKE["d"]
    a = rr.pcp.anchor_indices(n, rr.pcp.SPLIT_SEED, rr.pcp.HOLDOUT_FRACTION, n_anchors, rr.pcp.ANCHOR_DRAW_SEED)["anchor_idx"]
    panel = rr.pcp.knn_panel(X, a, k)
    fit = rr.ppf.fit_decoder(X, d, X.shape[1], 3)
    with torch.no_grad():
        z = fit["model"].encode(fit["x64"][torch.as_tensor(a, dtype=torch.long)])
    geo = rr.ppf.decoder_geometry(fit["curvature_model"], z)
    return {"X": X, "y": np.asarray(data["labels"]["lin"], float), "a": a, "panel": panel, "geo": geo, "d": d}


def test_probe_panel_matches_published_construction(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    y_hat = rr.runner._oof_predictions_for_label(s["X"], s["y"], 100.0, rr.pcp.N_OOF_FOLDS, rr.pcp.OOF_FOLD_SEED)
    loc = rr.pcp.local_r2_panel(s["y"], y_hat, s["panel"]["indices"], rr.pcp.MIN_FINITE_NEIGHBOURS)
    np.testing.assert_array_equal(pp["r2"], loc["r2"])
    ks = [kk for kk in rr.ppf.MULTISCALE_KS if kk <= s["panel"]["indices"].shape[1]]
    assert pp["Z_multi"].shape == (len(s["a"]), len(ks) + 2)


def test_extended_controls_and_partials(smoke):
    s = smoke
    pp = rr.probe_panel(s["X"], s["y"], s["a"], s["panel"], (100.0, 100.0))
    w, b0 = rr.global_probe(s["X"], s["y"], 100.0)
    sq = rr.split_quantities(s["X"], s["y"], s["a"], s["panel"], s["geo"], w, b0, s["d"])
    Z_ext = rr.extended_controls(pp["Z_multi"], sq["cols"], sq["roughness"])
    assert Z_ext.shape[1] == pp["Z_multi"].shape[1] + 2
    np.testing.assert_array_equal(Z_ext[:, -2], sq["cols"]["hess_label"])
    parts = rr.partials(sq["cols"], pp["r2"], Z_ext, 50)
    ref = rr.ppf.partial_row(sq["cols"][rr.MISMATCH], pp["r2"], Z_ext, 50)
    assert parts[rr.MISMATCH]["partial"] == ref["partial"] and set(parts) == {rr.MISMATCH, rr.ALIGN}


def test_overlap_blocks_count():
    rng = np.random.default_rng(0)
    neigh = np.array([rng.choice(500, 40, replace=False) for _ in range(120)])
    ov = rr.th.overlap_matrix(neigh)
    b = rr.overlap_blocks(ov, 16)
    assert b.shape == (120,) and len(np.unique(b)) == 16


def _synthetic_partial(n, rng):
    Z = rng.standard_normal((n, 2)); x = rng.standard_normal(n)
    r2 = 0.4 * x + Z @ np.array([0.5, -0.3]) + rng.standard_normal(n)
    return x, r2, Z


def test_cluster_bootstrap_covers_truth():
    rng = np.random.default_rng(1)
    truth = rr.pcp.controlled_partial(*_synthetic_partial(200000, rng))
    cover = 0
    for rep in range(200):
        x, r2, Z = _synthetic_partial(256, rng)
        blocks = rng.integers(0, 32, 256)
        ci = rr.cluster_bootstrap(x, r2, Z, blocks, 200, rep)
        cover += ci["lo"] <= truth <= ci["hi"]
    assert 0.90 <= cover / 200 <= 0.99


def test_bootstrap_skips_degenerate_replicates():
    x = np.full(40, np.nan); x[:3] = [1.0, 2.0, 3.0]
    r2 = np.arange(40.0); Z = np.ones((40, 1)); blocks = np.arange(40) % 8
    ci = rr.cluster_bootstrap(x, r2, Z, blocks, 50, 0)
    assert ci["n_ok"] + ci["n_skipped"] == 50 and ci["n_skipped"] > 0


def test_thinned_partial_uses_independent_anchors():
    rng = np.random.default_rng(2)
    neigh = np.array([rng.choice(2000, 30, replace=False) for _ in range(100)])
    ov = rr.th.overlap_matrix(neigh)
    x, r2, Z = _synthetic_partial(100, rng)
    t = rr.thinned_partial(x, r2, Z, ov, 0.10, 50)
    assert t["n_kept"] == int(rr.extract._indep(ov, 0.10).sum()) and "partial" in t


def test_heldout_positive_with_signal_zero_with_noise():
    rng = np.random.default_rng(3)
    n = 512
    Zb = rng.standard_normal((n, 3)); g = rng.standard_normal((n, 2)); blocks = np.arange(n) % 32
    r2_sig = Zb @ np.array([0.3, 0.2, 0.1]) + g @ np.array([0.8, -0.5]) + 0.5 * rng.standard_normal(n)
    r2_noise = Zb @ np.array([0.3, 0.2, 0.1]) + 0.5 * rng.standard_normal(n)
    sig = rr.heldout_delta_r2(r2_sig, Zb, g, blocks, 20, 0)
    noi = rr.heldout_delta_r2(r2_noise, Zb, g, blocks, 20, 0)
    assert sig["median"] > 0.05 and sig["frac_pos"] == 1.0
    assert abs(noi["median"]) < 0.02 and sig["n_splits"] == 20


def _arrays(n=64, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for v in rr.ns.VARIANTS:
        cv = rng.standard_normal((n, len(rr.ns.T_GRID)))
        out.update({f"{v}:r2_curve": cv, f"{v}:eq": rng.standard_normal(n), f"{v}:qq": rng.random(n) + 0.1,
                    f"{v}:t_star": rng.standard_normal(n), f"{v}:dR2": rng.standard_normal(n)})
    return out


def test_surrogate_fidelity_identical_curves():
    arr = _arrays(); arr["S:r2_curve"] = arr["S_model:r2_curve"].copy()
    s = rr.surrogate_fidelity(arr)
    assert s["spearman"] == pytest.approx(1.0) and s["median_abs_diff"] == 0.0 and s["n"] == 64


def test_cf_tables_use_published_readers(tmp_path):
    rng = np.random.default_rng(4)
    neigh = np.array([rng.choice(3000, 40, replace=False) for _ in range(64)])
    ov = rr.th.overlap_matrix(neigh)
    by_label = {"mag_r": _arrays(seed=1)}
    t = rr.cf_tables(by_label, ov, tmp_path)
    cv = by_label["mag_r"]["S_model:r2_curve"]; m = np.isfinite(by_label["mag_r"]["S_model:eq"])
    assert t["summary"]["mag_r"]["S_model"]["help"] == pytest.approx(float(np.mean(cv[m, 4] - cv[m, 2] > 0)))
    ov32 = ov.astype(np.float32).astype(float)       # the published thin npz stores the overlap as float32
    assert t["sign"]["mag_r"]["n"] == int((rr.extract._indep(ov32, rr.SIGN_THR) & np.isfinite(cv[:, 0])).sum())


def _write_published(tmp_path, partial=0.3):
    rows = [{"row": "environment"}]
    for d in (16, 20):
        rows.append({"row": "result", "d": d, "label": "mag_r",
                     "columns": {rr.MISMATCH: {"multiscale": {"partial": partial if d == 16 else 0.9, "p": 0.01}},
                                 rr.ALIGN: {"multiscale": {"partial": 0.1, "p": 0.2}}}})
    p = tmp_path / "split.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    arr = {f"mag_r:{k}": v for k, v in _arrays(seed=5).items()}
    c = tmp_path / "cf.npz"; np.savez(c, **arr)
    return p, c


def test_reference_uses_d16_rows_only(tmp_path):
    p, c = _write_published(tmp_path)
    ref = rr.published_reference(p, c)
    assert ref["split"][("mag_r", rr.MISMATCH)]["partial"] == 0.3


def test_guard_stops_on_perturbed_reference(tmp_path):
    p, c = _write_published(tmp_path)
    ref = rr.published_reference(p, c)
    ours = {"split": {k: dict(v) for k, v in ref["split"].items()}, "cf": json.loads(json.dumps(ref["cf"]))}
    assert rr.reproduction_diffs(ours, ref, "exact") == []
    ours["split"][("mag_r", rr.MISMATCH)]["partial"] += 0.01
    assert rr.reproduction_diffs(ours, ref, "refit") == []                      # within 0.02
    with pytest.raises(SystemExit, match=f"mag_r.*{rr.MISMATCH}"):
        rr.enforce_reproduction(ours, ref, "exact", labels=("mag_r",))
    ours["cf"]["mag_r"]["S_model"]["help"] += 1e-3
    assert any("help" in d for d in rr.reproduction_diffs(ours, ref, "refit"))


def test_refuses_geometry_sha_mismatch(tmp_path, monkeypatch):
    g = tmp_path / "g.npz"; np.savez(g, anchor_idx=np.arange(3))
    monkeypatch.setattr(sys, "argv", ["x", "--encoder", "vit_base", "--geometry-npz", str(g), "--geometry-sha256", "0" * 64,
                                      "--threads", "8", "--record-path", str(tmp_path / "r.jsonl")])
    with pytest.raises(SystemExit, match="sha256"):
        rr.main()
    assert not (tmp_path / "r.jsonl").exists()


def test_smoke_end_to_end(tmp_path, monkeypatch):
    rec = tmp_path / "11_review_robustness_smoke.jsonl"
    monkeypatch.setattr(sys, "argv", ["x", "--smoke", "--threads", "8", "--record-path", str(rec), "--n-perm", "20", "--n-boot", "30"])
    rr.main()
    rows = [json.loads(l) for l in rec.read_text().splitlines()]
    assert rows[0]["row"] == "environment"
    res = [r for r in rows if r["row"] == "result"]
    assert {(r["label"], r["alpha_mode"]) for r in res} == {(l, m) for l in ("lin", "nonlin_with_nan") for m in ("published", "tuned")}
    for r in res:
        assert np.isfinite(r["partials"]["published_controls"][rr.MISMATCH]["partial"])
        assert set(r["bootstrap"]) == {"16", "32", "64"} and "heldout" in r and "surrogate" in r and "cf" in r
    assert all(r["alpha"] == 100.0 for r in res if r["alpha_mode"] == "published")


rep = _load("11_review_robustness_report.py")


def _res(enc, lab, mode, mis):
    part = {rr.MISMATCH: {"partial": mis, "p": 0.001}, rr.ALIGN: {"partial": 0.2, "p": 0.01}}
    boot = {c: {"lo": mis - 0.1, "hi": mis + 0.1, "excludes_zero": True, "n_ok": 2000, "n_skipped": 0} for c in (rr.MISMATCH, rr.ALIGN)}
    return {"row": "result", "encoder": enc, "label": lab, "alpha_mode": mode, "alpha": 100.0 if mode == "published" else 3.7,
            "global_oof_r2": 0.6, "partials": {"published_controls": part, "extended_controls": part},
            "bootstrap": {"16": boot, "32": boot, "64": boot}, "thinned": {c: {"partial": 0.1, "p": 0.3, "n_kept": 22} for c in part},
            "heldout": {"median": 0.03, "p05": 0.01, "p95": 0.05, "frac_pos": 1.0, "n_splits": 20},
            "cf": {"S_model": {"help": 0.8, "hurt": 0.9, "t_star": 0.7}, "random_qmatched": {"help": 0.4, "hurt": 0.5}},
            "sign_test": {"n": 15, "help": 0.8, "p_help": 0.01}, "surrogate": {"spearman": 0.95, "median_abs_diff": 0.001, "n": 512}}


def test_report_sections(tmp_path):
    rows = [{"row": "environment", "encoder": "vit_base", "numpy": "2.5.1"}, {"row": "guard", "encoder": "vit_base", "mode": "exact", "passed": True}]
    rows += [_res("vit_base", "mag_r", m, -0.3) for m in ("published", "tuned")] + [_res("vit_base", "mag_r", "tuned", -0.4)]
    p = tmp_path / "r.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    data = rep.load([p])
    assert len(data["rows"]) == 2
    rep.write_report(data, tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    for h in ("## Concern 2", "## Concern 3", "## Concern 4", "## Concern 5", "## Limitation", "guard: exact PASS"):
        assert h in text, h
    assert "-0.400" in text      # the re-run tuned row won



def test_guard_refuses_missing_or_incomplete_reference(tmp_path):
    p, c = _write_published(tmp_path)
    with pytest.raises(SystemExit, match="published reference missing"):
        rr.published_reference(tmp_path / "nope.jsonl", c)
    ref = rr.published_reference(p, c)
    ours = {"split": {k: dict(v) for k, v in ref["split"].items()}, "cf": json.loads(json.dumps(ref["cf"]))}
    with pytest.raises(SystemExit, match="photo_z"):          # the reference holds mag_r only
        rr.enforce_reproduction(ours, ref, "exact")


def test_guard_summary_counts_and_max_diff(tmp_path):
    p, c = _write_published(tmp_path)
    ref = rr.published_reference(p, c)
    ours = {"split": {k: dict(v) for k, v in ref["split"].items()}, "cf": json.loads(json.dumps(ref["cf"]))}
    ours["split"][("mag_r", rr.MISMATCH)]["partial"] += 0.005
    s = rr.enforce_reproduction(ours, ref, "refit", labels=("mag_r",))
    assert s["n_split"] == 2 and s["n_cf"] > 0
    assert s["max_abs_diff_split"] == pytest.approx(0.005) and s["max_abs_diff_cf"] == 0.0


def _res_full(enc, lab, mode, mis, lo, hi, alpha=100.0):
    r = _res(enc, lab, mode, mis)
    r["alpha"] = alpha; r["fold_alphas"] = [0.1] * 5
    r["bootstrap"]["32"][rr.MISMATCH].update({"lo": lo, "hi": hi, "excludes_zero": lo > 0 or hi < 0})
    r["cf"]["random_qmatched"]["help"] = 0.07; r["cf"]["S_model"]["t_star"] = 0.61
    return r


def test_report_discloses_alpha_edge_and_dependence_at_alpha_star(tmp_path):
    rows = [{"row": "environment", "encoder": "vit_base", "alpha_grid": list(rr.ALPHA_GRID)},
            {"row": "guard", "encoder": "vit_base", "mode": "exact", "passed": True, "n_split": 8, "n_cf": 40,
             "max_abs_diff_split": 0.0, "max_abs_diff_cf": 0.0}]
    rows += [_res_full("vit_base", "stellar_mass", "published", -0.04, -0.2, 0.1),
             _res_full("vit_base", "stellar_mass", "tuned", -0.132, -0.274, 0.077, alpha=min(rr.ALPHA_GRID))]
    p = tmp_path / "r.jsonl"; p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    rep.write_report(rep.load([p]), tmp_path)
    text = (tmp_path / "REPORT.md").read_text()
    assert "grid edge" in text and "0.1 x5" in text
    assert "anchor-level permutation" in text
    assert "| vit_base | stellar_mass | alpha* | hess_mismatch_emp | -0.132 | [-0.274, +0.077] |" in text
    assert "8 split cells and 40 counterfactual values" in text and "max |diff| 0" in text
    assert "0.07" in text and "0.61" in text                     # random null help and t*
    assert "appendix" in text.lower() and "|local R2(surrogate) - local R2(probe)|" in text
