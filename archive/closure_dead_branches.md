# Stage 6: dead flags and branches removed

Removed blocks, verbatim, from Stage 6 (Task 8). Each heading gives the file and the original
line range at the time of removal. See the Step 1 grep transcript in the Task 8 report for how
each candidate was found and adjudicated.

## docs/latex/ml4ps/appendix_gen.py:71-95 (INCLUDE_RELATIVE_II is a never-reassigned module constant fixed to False; no invocation of appendix_gen.py can change it, so the guarded Appendix X block and the `rrows` computation feeding it never ran)

```python
# --- C: relative II (two-embedding alignment pilot; off since the 2026-09-18 reframe dropped the cross-survey
#         application from the manuscript. Set True to regenerate it.)
INCLUDE_RELATIVE_II = False
runs = [("$d=20$, seed 0", C + "08_relative_ii_d20.jsonl"), ("$d=25$, seed 0", C + "08_relative_ii_d25_seed0.jsonl"), ("$d=20$, seed 1", C + "08_relative_ii_d20_seed1.jsonl")]
rrows = []
for name, f in runs:
    for r in rows(f):
        if r.get("row") == "result" and r["mknn_k"] == 20:
            c = r["columns"]
            rrows.append(f"{name} & {r['align_r2_holdout']:.2f} & {c['tan_resid']['median']:.2f} & " + " & ".join(cell(c[k]["multiscale"]) for k in ("H_tan_F", "H_tan_G", "tan_resid", "II_rel", "II_rel_loc", "II_rel_emp")) + r" \\")
if rrows and INCLUDE_RELATIVE_II:
    out.append(r"""\section*{Appendix X: relative second fundamental form, pilot}  % disabled block; reletter before re-enabling
Two sphere-projected decoders (HSC $=F$, Legacy $=G$; Phase 7 protocol), a global ridge map $A$ from $x_F$ to $x_G$ (fit on the 8{,}000 training rows), 2{,}048 seeded anchors. Columns: holdout $R^2$ of $A$; median first-order obstruction $\norm{AJ_F - J_G L}/\norm{AJ_F}$ with $L = J_G^{+}AJ_F$; then multi-scale density-controlled partials against MKNN ($k=20$; log radius at $k\in\{10,30,100,300\}$ in both spaces) of each decoder's $\norm{\Htan}$, the first-order obstruction, $\norm{\mathrm{II}_G(L\cdot,L\cdot) - P_N^G A\,\mathrm{II}_F}$ (because first-order matching $AJ_F = J_G$ fails at the 44\% level, the second derivative of the alignment residual is not coordinate invariant; in the decoder chart its target-normal quadratic coefficient additionally contains $-P_N^G A J_F \Gamma^F$, so relative II is reported only as an extrinsic diagnostic), the same with $L$ fitted on 256 neighbours' latent codes, and a decoder-free estimate from the quadratic coefficient of the alignment residual on $F$'s tangent coordinates.
\begin{table}[h]
\centering\footnotesize\setlength{\tabcolsep}{3.5pt}
\begin{tabular}{lcccccccc}
\toprule
run & $R^2_A$ & tan.\ resid & $\norm{\Htan}$ HSC & $\norm{\Htan}$ Legacy & tan.\ resid & $\mathrm{II}_{\mathrm{rel}}$ & $\mathrm{II}_{\mathrm{rel}}$ (local $L$) & $\mathrm{II}_{\mathrm{rel}}$ (data) \\
\midrule""")
    out += rrows
    out.append(r"""\bottomrule
\end{tabular}
\caption{Relative-II pilot. $^{*}$ not significant at 0.05.}
\label{tab:relii}
\end{table}""")
```

## docs/latex/ml4ps/appendix_gen.py:191 (diagnostic print statement referenced `rrows`, which was only defined by the removed Appendix C block; fixed the resulting `NameError` caught by the Step 3 gate's generator run -- this is a repair of breakage from the removal above, not itself dead code)

Before:
```python
open(p, "w").write(s); print("appendix spliced:", len(out), "lines;", f"{len(have)} encoders;", f"{len(erows)} cf rows;", "xfit" if xf else "no-xfit", "alpha" if al else "no-alpha", len(rrows), "relii rows")
```
After:
```python
open(p, "w").write(s); print("appendix spliced:", len(out), "lines;", f"{len(have)} encoders;", f"{len(erows)} cf rows;", "xfit" if xf else "no-xfit", "alpha" if al else "no-alpha")
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:86 (`--mode swiss-roll` is the only caller of `make_swiss_roll` in this runner)

```python
from sklearn.datasets import make_swiss_roll  # noqa: E402
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:175 (`SWISS_RANK_RHO_PASS` only read by `run_swiss_roll`, the removed `--mode swiss-roll` dispatch target)

```python
SWISS_RANK_RHO_PASS = 0.5
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:191-192 (`SWISS` and `SWISS_EPOCHS` only read by `run_swiss_roll`; no other runner imports `adj.SWISS` or `adj.SWISS_EPOCHS` -- confirmed by grepping `adj\.` across the sibling `09_*` runners)

```python
SWISS = {"n": 3000, "random_state": 0, "d": 2, "k": 256, "n_anchors": 256}
SWISS_EPOCHS = 300
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:331-365 (`fit_ambient_field_at_anchors` is only called from `run_swiss_roll`)

```python
# --- our instrument on the Swiss roll (ambient, no sphere projection) ---------------------


def fit_ambient_field_at_anchors(X: np.ndarray, d: int, anchor_idx: np.ndarray, max_epochs: int) -> Dict[str, Any]:
    """The same sealed calls as ``runner.fit_and_field_at_anchors`` (PlainAutoEncoder with the
    frozen hidden/activation/train_cfg/seeds, split_indices, train_plain_ae, plain_decoder_curvature
    at the anchor codes only) MINUS the SphereProjectedDecoder wrapper, which that function applies
    unconditionally under ``pcp.DECODER_IMAGE_PROJECTION == "sphere"``. The Swiss roll is not on a
    sphere, so its curvature is ambient. Mirrored here rather than edited there."""
    torch.manual_seed(pcp.TORCH_INIT_SEED)
    model = cae.PlainAutoEncoder(in_dim=X.shape[1], latent_dim=d, hidden=pcp.AE_HIDDEN, activation=pcp.AE_ACTIVATION)
    train_idx, holdout_idx = crossmodal_curvature.split_indices(X.shape[0], pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION)
    x32 = torch.tensor(X, dtype=torch.float32)
    x64 = torch.tensor(X, dtype=torch.float64)
    cfg = dict(pcp.TRAIN_CFG)
    cfg["max_epochs"] = max_epochs
    t0 = time.monotonic()
    cae.train_plain_ae(model, x32[torch.as_tensor(train_idx, dtype=torch.long)], cfg)
    wallclock_fit_s = time.monotonic() - t0
    model.eval().double()
    x_holdout64 = x64[torch.as_tensor(holdout_idx, dtype=torch.long)]
    x_anchor64 = x64[torch.as_tensor(np.asarray(anchor_idx), dtype=torch.long)]
    with torch.no_grad():
        z_anchor = model.encode(x_anchor64)
        y_holdout = model(x_holdout64)["y"]
        image = model.decode(z_anchor).numpy()
    recon = cae.reconstruction_stats(x_holdout64, y_holdout)
    sig = float((torch.linalg.norm(x_holdout64, dim=1) ** 2).mean())
    field = decoder_curvature.plain_decoder_curvature(model, z_anchor)
    return {
        "H_vec": field["H_vec"].numpy(), "image": image,
        "var_explained": 1.0 - recon["mse_total"] / sig, "wallclock_fit_s": wallclock_fit_s,
    }
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:402-457 (`run_swiss_roll`: dispatch target of the removed `--mode swiss-roll` choice; no paper invocation in `test_paper_invocations.py` uses this mode)

```python
# --- mode: swiss-roll ------------------------------------------------------------------------


def run_swiss_roll(args: argparse.Namespace, record_path: Path) -> bool:
    print("\n" + "=" * 78 + "\nLOW-d ANCHOR: Swiss roll (d=2 in R^3), analytic mean curvature\n" + "=" * 78)
    print(f"DECISION RULE (fixed before the numbers): rank Spearman rho vs analytic truth >= {SWISS_RANK_RHO_PASS} "
          f"-> SWISS ROLL PASS, per instrument. Coarse anchor only.")
    n, d, k, n_anchors = SWISS["n"], SWISS["d"], SWISS["k"], SWISS["n_anchors"]
    X_raw, t = make_swiss_roll(n_samples=n, noise=0.0, random_state=SWISS["random_state"])
    s = float(X_raw.std())
    X = ((X_raw - X_raw.mean(axis=0)) / s).astype(np.float64)
    H_true_norm = curvature_probe.swiss_roll_analytic_H_scaled(t, s)
    H_true_vec = decoder_curvature.swiss_roll_analytic_H_vector(t, s)

    split = pcp.anchor_indices(n, pcp.SPLIT_SEED, pcp.HOLDOUT_FRACTION, n_anchors, pcp.ANCHOR_DRAW_SEED)
    anchor_idx = split["anchor_idx"]
    panel = pcp.knn_panel(X, anchor_idx, k)
    rr = measure_r_over_R(X, panel["distances"][:, -1])
    print(f"n={n} d={d} k={k} (rule: largest preset <= n/8={n // 8} -> 256) anchors={n_anchors} "
          f"r_knn={rr['r_knn']:.4f} R={rr['R']:.4f} r/R={rr['r_over_R']:.4f}")
    print("Ours: PlainAutoEncoder 3->2, frozen hidden/activation/train_cfg/seeds, AMBIENT curvature -- no sphere "
          "projection, the roll is not on a sphere.")

    ours = fit_ambient_field_at_anchors(X, d, anchor_idx, args.max_epochs if args.max_epochs is not None else SWISS_EPOCHS)
    ours_axes = _fidelity_axes(ours["H_vec"], H_true_vec[anchor_idx])
    ours_axes["rho_vs_log_knn_radius"] = _spearman(np.linalg.norm(ours["H_vec"], axis=1), panel["log_knn_radius"])
    print(f"[ours] var_explained={ours['var_explained']:.4f} fit {ours['wallclock_fit_s']:.1f}s")

    truth_rho_radius = _spearman(H_true_norm[anchor_idx], panel["log_knn_radius"])
    rows = [
        {"instrument": "ours_H_ambient", "rank_spearman_rho": ours_axes["rank_spearman_rho"],
         "direction_median_cosine": ours_axes["direction_median_cosine"], "magnitude_median_ratio": ours_axes["magnitude_median_ratio"],
         "magnitude_ratio_cv": ours_axes["magnitude_ratio_cv"], "calibration_slope": ours_axes["calibration_slope"],
         "calibration_r2": ours_axes["calibration_r2"], "rho_vs_log_knn_radius": ours_axes["rho_vs_log_knn_radius"]},
        {"instrument": "truth", "rho_vs_log_knn_radius": truth_rho_radius},
    ]
    print()
    _print_table(rows, ["instrument", "rank_spearman_rho", "direction_median_cosine", "magnitude_median_ratio",
                        "magnitude_ratio_cv", "calibration_slope", "calibration_r2", "rho_vs_log_knn_radius"])
    print()

    verdicts = {}
    for name, rho in (("ours", ours_axes["rank_spearman_rho"]),):
        ok = rho is not None and np.isfinite(rho) and rho >= SWISS_RANK_RHO_PASS
        verdicts[name] = ok
        print(f"rank rho={_fmt(rho)} >= {SWISS_RANK_RHO_PASS}: {'PASS' if ok else 'FAIL'}   ->  SWISS ROLL {'PASS' if ok else 'FAIL'} [{name}]")
    print(f"ours direction median cosine={_fmt(ours_axes['direction_median_cosine'])} (reported beside rank, not gated at this anchor)")

    base = {"experiment": EXPERIMENT, "mode": "swiss-roll", "noise": "0", "timestamp": _utc_now(), "n": n, "d": d, "k": k,
            "n_anchors": n_anchors, **{f"regime_{k_}": v for k_, v in rr.items()}, "decision_rule": f"rank_rho >= {SWISS_RANK_RHO_PASS}",
            "truth_rho_vs_log_knn_radius": truth_rho_radius}
    _append({**base, "instrument": "ours_H_ambient", "var_explained": ours["var_explained"], "max_epochs": args.max_epochs or SWISS_EPOCHS,
             "scores": ours_axes, "verdict": "PASS" if verdicts["ours"] else "FAIL"}, record_path)
    return True
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:88, 90 (`cae` and `crossmodal_curvature` imports; dead only as a consequence of removing `fit_ambient_field_at_anchors`, their sole caller in this file -- caught by `unused_symbols.py` after the branch removal)

```python
from pu_manifold import cae  # noqa: E402
from pu_manifold import crossmodal_curvature  # noqa: E402
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:561 (`--mode` choices: `"swiss-roll"` removed from the tuple; `"smoke"` and `"sphere-fixture"` kept -- `test_paper_invocations.py` only exercises `sphere-fixture`)

Before:
```python
    p.add_argument("--mode", choices=["smoke", "swiss-roll", "sphere-fixture"], required=True)
```
After:
```python
    p.add_argument("--mode", choices=["smoke", "sphere-fixture"], required=True)
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:568 (`--max-epochs` help text: dropped the swiss-roll-only clause referencing the removed `SWISS_EPOCHS`)

Before:
```python
    p.add_argument("--max-epochs", type=int, default=None,
                   help=f"AE epoch budget; default frozen MAX_EPOCHS={pcp.MAX_EPOCHS} for sphere-fixture, {SWISS_EPOCHS} for swiss-roll")
```
After:
```python
    p.add_argument("--max-epochs", type=int, default=None,
                   help=f"AE epoch budget; default frozen MAX_EPOCHS={pcp.MAX_EPOCHS} for sphere-fixture")
```

## notebooks/diagnostics/09_instrument_adjudication_run.py:585-586 (`main()` dispatch branch for the removed `--mode swiss-roll`)

```python
    if args.mode == "swiss-roll":
        sys.exit(0 if run_swiss_roll(args, record_path) else 1)
```
