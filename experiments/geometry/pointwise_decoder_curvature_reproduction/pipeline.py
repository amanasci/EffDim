"""Bounded R1–R4 reproduction. No Q. No new sphere training. 60-minute wall."""

from __future__ import annotations

import json
import platform
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import make_swiss_roll

_EXP = Path(__file__).resolve().parents[2]
_REPO = Path(__file__).resolve().parents[3]
_NB = _REPO / "notebooks"
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))
if str(_NB) not in sys.path:
    sys.path.insert(0, str(_NB))

from pu_manifold import cae, curvature_probe, decoder_curvature

from geometry.pointwise_decoder_curvature_reproduction import vendor_varying_ii_controls as vic
from geometry.pointwise_decoder_curvature_reproduction.config import (
    AE_ACTIVATION,
    AE_HIDDEN,
    COLLEAGUE,
    CURVATURE_EXPERIMENTS_SHA,
    F_DIFFERENTIATED_AFTER_NORMALIZE,
    FIXTURE_VALIDITY_AUDIT_SHA,
    H_IS_AVERAGED,
    HISTORICAL,
    II_REMOVES_SPHERE_RADIAL,
    OUT_REL,
    RESERVE_WRITE_S,
    SWEEP_CFG,
    SWEEP_EPOCHS,
    SWEEP_N,
    SWEEP_SEED,
    SWEEP_TORCH_INIT,
    SWISS_ANCHOR_DRAW_SEED,
    SWISS_CFG,
    SWISS_D,
    SWISS_EPOCHS,
    SWISS_HOLDOUT_FRACTION,
    SWISS_N,
    SWISS_N_ANCHORS,
    SWISS_RANDOM_STATE,
    SWISS_SPLIT_SEED,
    SWISS_TORCH_INIT,
    TOL,
    WALL_S,
)
from geometry.pointwise_decoder_curvature_reproduction.formulas import (
    anchor_indices,
    axes,
    autodiff_jets,
    decompose_hessians,
    run_unit_tests,
    split_indices,
)
from geometry.pointwise_decoder_curvature_reproduction.reports import (
    write_all_reports,
    write_protocol_docs,
)


@dataclass
class ExpConfig:
    output_dir: str = OUT_REL
    wall_s: float = WALL_S
    device: str = "cpu"
    skip_r5: bool = True


def _remaining(t0: float, wall: float) -> float:
    return wall - RESERVE_WRITE_S - (time.time() - t0)


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n")


def _var_explained(x64: torch.Tensor, y64: torch.Tensor) -> float:
    rec = cae.reconstruction_stats(x64, y64)
    sig = float((torch.linalg.norm(x64, dim=1) ** 2).mean())
    return 1.0 - rec["mse_total"] / sig


def _train_plain(X: np.ndarray, d: int, cfg: dict, torch_init: int, train_idx=None) -> tuple:
    D = X.shape[1]
    torch.manual_seed(int(torch_init))
    model = cae.PlainAutoEncoder(in_dim=D, latent_dim=d, hidden=AE_HIDDEN, activation=AE_ACTIVATION)
    x32 = torch.tensor(X, dtype=torch.float32)
    if train_idx is None:
        x_train = x32
    else:
        x_train = x32[torch.as_tensor(np.asarray(train_idx), dtype=torch.long)]
    t1 = time.time()
    model.train().float()
    info = cae.train_plain_ae(model, x_train, dict(cfg))
    t_train = time.time() - t1
    model.eval().double()
    return model, t_train, dict(info)


def _decode_curvature(model, z: torch.Tensor) -> tuple[np.ndarray, float]:
    t2 = time.time()
    field = decoder_curvature.plain_decoder_curvature(model, z)
    t_curv = time.time() - t2
    return field["H_vec"].detach().cpu().numpy(), t_curv


def _parity_row(cell: str, hist: dict, new: dict) -> dict:
    def ad(key_h, key_n):
        hv, nv = hist.get(key_h), new.get(key_n)
        if hv is None or nv is None or not np.isfinite(hv) or not np.isfinite(nv):
            return float("nan")
        return float(abs(nv - hv))

    rho_tol = TOL["rho_r1"] if cell == "R1" else TOL["rho"]
    row = {
        "cell": cell,
        "fixture": hist["fixture"],
        "d": hist["d"],
        "D": hist["D"],
        "n": hist["n"],
        "historical_var_explained": hist.get("var_explained"),
        "new_var_explained": new.get("var_explained"),
        "abs_diff_var_explained": ad("var_explained", "var_explained"),
        "historical_rho": hist["rho"],
        "new_rho": new.get("rho"),
        "abs_diff_rho": ad("rho", "rho"),
        "historical_cosine": hist["cosine"],
        "new_cosine": new.get("median_cosine"),
        "abs_diff_cosine": ad("cosine", "median_cosine"),
        "historical_ratio": hist["ratio"],
        "new_ratio": new.get("median_ratio"),
        "abs_diff_ratio": ad("ratio", "median_ratio"),
        "t_train_s": new.get("t_train_s"),
        "t_curv_s": new.get("t_curv_s"),
        "seed_torch_init": new.get("seed_torch_init"),
        "seed_data": new.get("seed_data"),
        "max_epochs": new.get("max_epochs"),
        "checkpoint": new.get("checkpoint"),
        "backend": new.get("backend"),
    }
    checks = []
    if hist.get("var_explained") is not None and np.isfinite(row["abs_diff_var_explained"]):
        checks.append(row["abs_diff_var_explained"] <= TOL["var_explained"])
        row["pass_var_explained"] = bool(checks[-1])
    else:
        row["pass_var_explained"] = None
    checks.append(bool(np.isfinite(row["abs_diff_rho"]) and row["abs_diff_rho"] <= rho_tol))
    row["pass_rho"] = checks[-1]
    checks.append(bool(np.isfinite(row["abs_diff_cosine"]) and row["abs_diff_cosine"] <= TOL["cosine"]))
    row["pass_cosine"] = checks[-1]
    checks.append(bool(np.isfinite(row["abs_diff_ratio"]) and row["abs_diff_ratio"] <= TOL["ratio"]))
    row["pass_ratio"] = checks[-1]
    row["pass_all_available"] = all(checks)
    row["rho_tolerance_used"] = rho_tol
    return row


def _decompose_points(model, z: torch.Tensor, n_max: int, sphere: bool) -> pd.DataFrame:
    decode_one = decoder_curvature.plain_decoder_map(model)
    n = min(int(z.shape[0]), n_max)
    rows = []
    for i in range(n):
        x, J, Q = autodiff_jets(decode_one, z[i])
        dcmp = decompose_hessians(x, J, Q, sphere=sphere)
        rows.append(
            {
                "i": i,
                "sphere": sphere,
                "energy_II_E": dcmp["energy_II_E"],
                "energy_II_R": dcmp["energy_II_R"] if sphere else float("nan"),
                "energy_B_S": dcmp["energy_B_S"] if sphere else float("nan"),
                "f_res": dcmp["f_res"] if sphere else float("nan"),
                "H_E_unavg_norm": float(np.linalg.norm(dcmp["H_E_unavg"])),
                "H_E_avg_norm": float(np.linalg.norm(dcmp["H_E_avg"])),
                "H_S_unavg_norm": float(np.linalg.norm(dcmp["H_S_unavg"])) if sphere else float("nan"),
                "reconstruct_rel": dcmp["reconstruct_rel"] if sphere else float("nan"),
                "PT_II_E": dcmp["PT_II_E"],
            }
        )
    return pd.DataFrame(rows)


def _persist(out: Path, cells, parity, point_rows, decomp_frames, radial_rows) -> None:
    if cells:
        pd.DataFrame(cells).to_csv(out / "reproduction_cells.csv", index=False)
    if parity:
        pd.DataFrame(parity).to_csv(out / "metric_parity.csv", index=False)
    if point_rows:
        pd.concat(point_rows, ignore_index=True).to_parquet(out / "pointwise_metrics.parquet", index=False)
    if decomp_frames:
        pd.concat(decomp_frames, ignore_index=True).to_parquet(out / "curvature_decomposition.parquet", index=False)
    if radial_rows:
        pd.DataFrame(radial_rows).to_csv(out / "radial_baseline.csv", index=False)


def run(cfg: ExpConfig) -> dict:
    t0 = time.time()
    root = _REPO
    out = (root / cfg.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    skipped: list[str] = []
    stages: list[str] = []
    write_protocol_docs(out)

    env = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "cuda_available": bool(torch.cuda.is_available()),
        "device": cfg.device,
        "cudnn": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
        "deterministic": False,
        "curvature_experiments_sha": CURVATURE_EXPERIMENTS_SHA,
        "fixture_validity_audit_sha": FIXTURE_VALIDITY_AUDIT_SHA,
        "colleague": COLLEAGUE,
    }
    _write_json(out / "environment.json", env)

    print("[repro] unit tests", flush=True)
    tests = run_unit_tests()
    _write_json(out / "unit_test_results.json", tests)
    if not tests["all_passed"]:
        _write_json(out / "COMPLETE.json", {"status": "blocked", "reason": "unit tests failed", "tests": tests})
        pd.DataFrame().to_csv(out / "reproduction_cells.csv", index=False)
        pd.DataFrame().to_csv(out / "metric_parity.csv", index=False)
        pd.DataFrame([{"status": "blocked"}]).to_csv(out / "radial_baseline.csv", index=False)
        pd.DataFrame({"cell": [], "sample_id": []}).to_parquet(out / "pointwise_metrics.parquet", index=False)
        pd.DataFrame({"cell": []}).to_parquet(out / "curvature_decomposition.parquet", index=False)
        write_all_reports(out, blocked=True, tests=tests, env=env)
        return {"blocked": True, "tests": tests}
    stages.append("unit_tests")

    cells = []
    parity = []
    point_rows = []
    decomp_frames = []
    radial_rows = []

    def maybe_stop(tag: str) -> bool:
        if _remaining(t0, cfg.wall_s) < 20:
            skipped.append(f"{tag}:wall")
            return True
        return False

    def persist() -> None:
        _persist(out, cells, parity, point_rows, decomp_frames, radial_rows)
        _write_json(
            out / "runtime.json",
            {
                "runtime_s": time.time() - t0,
                "wall_s": cfg.wall_s,
                "stages": list(stages),
                "skipped": [s for s in skipped if s],
                "n_new_decoders": len([c for c in stages if c.startswith("R")]),
            },
        )

    try:
        # ---- R1 Swiss roll (ambient decoder; no Q) ----
        if not maybe_stop("R1"):
            print("[repro] R1 swiss roll", flush=True)
            X_raw, t = make_swiss_roll(n_samples=SWISS_N, noise=0.0, random_state=SWISS_RANDOM_STATE)
            s = float(X_raw.std())
            X = ((X_raw - X_raw.mean(axis=0)) / s).astype(np.float64)
            H_true_vec = decoder_curvature.swiss_roll_analytic_H_vector(t, s)
            split = anchor_indices(SWISS_N, SWISS_SPLIT_SEED, SWISS_HOLDOUT_FRACTION, SWISS_N_ANCHORS, SWISS_ANCHOR_DRAW_SEED)
            model, t_train, info = _train_plain(X, SWISS_D, SWISS_CFG, SWISS_TORCH_INIT, train_idx=split["train_idx"])
            x64 = torch.tensor(X, dtype=torch.float64)
            with torch.no_grad():
                z_anchor = model.encode(x64[torch.as_tensor(split["anchor_idx"])])
                y_ho = model(x64[torch.as_tensor(split["holdout_idx"])])["y"]
            ve = _var_explained(x64[torch.as_tensor(split["holdout_idx"])], y_ho)
            H_est, t_curv = _decode_curvature(model, z_anchor)
            ax = axes(H_est, H_true_vec[split["anchor_idx"]])
            rec = {
                "cell": "R1",
                "var_explained": ve,
                "rho": ax["rho"],
                "median_cosine": ax["median_cosine"],
                "median_ratio": ax["median_ratio"],
                "t_train_s": t_train,
                "t_curv_s": t_curv,
                "seed_torch_init": SWISS_TORCH_INIT,
                "seed_data": SWISS_RANDOM_STATE,
                "max_epochs": SWISS_EPOCHS,
                "epochs_run": info.get("epochs_run"),
                "checkpoint": "in-memory (no .pt written historically)",
                "backend": f"torch {torch.__version__} cpu",
                "n_eval": int(len(split["anchor_idx"])),
                "sample_ids": [int(i) for i in split["anchor_idx"][:16]],
            }
            cells.append(rec)
            parity.append(_parity_row("R1", HISTORICAL["R1"], rec))
            point_rows.append(pd.DataFrame({"cell": "R1", "sample_id": split["anchor_idx"], "H_est_norm": np.linalg.norm(H_est, axis=1), "H_true_norm": np.linalg.norm(H_true_vec[split["anchor_idx"]], axis=1)}))
            decomp_frames.append(_decompose_points(model, z_anchor[: min(16, z_anchor.shape[0])], 16, sphere=False).assign(cell="R1"))
            stages.append("R1")
            persist()
            print(f"[repro] R1 ve={ve:.4f} rho={ax['rho']:.4f} cos={ax['median_cosine']:.4f} ratio={ax['median_ratio']:.4f} t={t_train:.1f}+{t_curv:.1f}s", flush=True)

        # ---- R2–R4 cubic/ridge sweep protocol. R5 would be a 5th decoder (hard cap = 4). ----
        primary = [("R2", "cubic", 28), ("R3", "ridge", 28), ("R4", "ridge", 768)]
        skipped.append("R5:hard_cap_max_4_new_decoders")

        for cell, fixture, D in primary:
            if maybe_stop(cell):
                persist()
                continue
            if cell == "R4" and _remaining(t0, cfg.wall_s) < 700:
                skipped.append("R4:insufficient_remaining_for_D768")
                persist()
                continue
            print(f"[repro] {cell} {fixture} d=16 D={D}", flush=True)
            fx = vic.FAMILIES[fixture](SWEEP_N, 16, D, SWEEP_SEED)
            X = np.asarray(fx["X"], dtype=np.float64)
            H_true = np.asarray(fx["H_vec"], dtype=np.float64)
            ids = np.arange(SWEEP_N, dtype=np.int64)
            model, t_train, info = _train_plain(X, 16, SWEEP_CFG, SWEEP_TORCH_INIT, train_idx=None)
            x64 = torch.tensor(X, dtype=torch.float64)
            with torch.no_grad():
                rec_y = model(x64)["y"]
                z = model.encode(x64)
            ve = _var_explained(x64, rec_y)
            H_est, t_curv = _decode_curvature(model, z)
            ax = axes(H_est, H_true)
            rec = {
                "cell": cell,
                "var_explained": ve,
                "rho": ax["rho"],
                "median_cosine": ax["median_cosine"],
                "median_ratio": ax["median_ratio"],
                "t_train_s": t_train,
                "t_curv_s": t_curv,
                "seed_torch_init": SWEEP_TORCH_INIT,
                "seed_data": SWEEP_SEED,
                "max_epochs": SWEEP_EPOCHS,
                "epochs_run": info.get("epochs_run"),
                "checkpoint": "in-memory (07 sweep wrote JSONL only)",
                "backend": f"torch {torch.__version__} cpu",
                "n_eval": int(SWEEP_N),
                "ii_cv": fx["ii_variation"]["hess_fro_cv"],
            }
            cells.append(rec)
            parity.append(_parity_row(cell, HISTORICAL[cell], rec))
            point_rows.append(pd.DataFrame({"cell": cell, "sample_id": ids, "H_est_norm": np.linalg.norm(H_est, axis=1), "H_true_norm": np.linalg.norm(H_true, axis=1)}))
            n_de = 16 if D >= 768 else 32
            decomp_frames.append(_decompose_points(model, z[:n_de], n_de, sphere=False).assign(cell=cell))
            stages.append(cell)
            persist()
            print(f"[repro] {cell} ve={ve:.6f} rho={ax['rho']:.4f} cos={ax['median_cosine']:.4f} ratio={ax['median_ratio']:.4f} t={t_train:.1f}+{t_curv:.1f}s", flush=True)
    except Exception as exc:
        skipped.append(f"exception:{type(exc).__name__}:{exc}")
        persist()
        print(f"[repro] exception {exc}\n{traceback.format_exc()}", flush=True)

    cache_dirs = [
        root / "notebooks" / ".cache",
        Path.home() / "Documents" / "Projects" / "EffDim" / "notebooks" / ".cache",
    ]
    pt_hits = []
    for d in cache_dirs:
        if d.is_dir():
            pt_hits.extend(d.glob("**/*86471*.pt"))
            pt_hits.extend(d.glob("**/*sphere*.pt"))
    if not pt_hits:
        skipped.append("cached_sphere_evaluation_skipped_missing_weights")
    else:
        skipped.append("cached_sphere_evaluation_skipped_by_resource_cap")
    radial_rows.append({"fixture": "none_primary_sphere", "status": "radial_baseline_not_applicable", "reason": "R1–R4 outputs are not constrained to the unit sphere"})

    runtime_s = time.time() - t0
    qtab = pd.DataFrame(parity)
    if point_rows:
        pd.concat(point_rows, ignore_index=True).to_parquet(out / "pointwise_metrics.parquet", index=False)
    if decomp_frames:
        pd.concat(decomp_frames, ignore_index=True).to_parquet(out / "curvature_decomposition.parquet", index=False)
    pd.DataFrame(cells).to_csv(out / "reproduction_cells.csv", index=False)
    qtab.to_csv(out / "metric_parity.csv", index=False)
    pd.DataFrame(radial_rows).to_csv(out / "radial_baseline.csv", index=False)

    primary_done = [c for c in ("R1", "R2", "R3", "R4") if c in stages]
    n_pass = int(qtab["pass_all_available"].sum()) if len(qtab) else 0
    cubic_ok = bool(((qtab.cell == "R2") & qtab.pass_all_available).any()) if len(qtab) else False
    ridge_ok = bool((qtab.cell.isin(["R3", "R4"]) & qtab.pass_all_available).any()) if len(qtab) else False

    if n_pass >= 3 and cubic_ok and ridge_ok:
        primary_label = "colleague_decoder_results_reproduced"
    elif len(primary_done) >= 1 and n_pass >= 1:
        primary_label = "partial_reproduction_with_protocol_drift"
    elif len(primary_done) >= 1:
        primary_label = "reported_decoder_results_not_reproduced"
    else:
        primary_label = "reproduction_blocked_by_missing_definition_or_code"

    estimand = "reported_quantity_is_full_euclidean_curvature"
    radial = "radial_baseline_not_applicable"

    decision = {
        "primary_reproduction_label": primary_label,
        "secondary_estimand_label": estimand,
        "radial_baseline_label": radial,
        "H_is_averaged_1_over_d": H_IS_AVERAGED,
        "II_removes_sphere_radial": II_REMOVES_SPHERE_RADIAL,
        "F_differentiated_after_normalization": F_DIFFERENTIATED_AFTER_NORMALIZE,
        "historical_H": "g^{ab} II_ab  (UNNORMALIZED trace)",
        "historical_II": "(I - P_T) D^2 F",
        "n_primary_cells_passed": n_pass,
        "primary_done": primary_done,
        "skipped": [s for s in skipped if s],
        "colleague": COLLEAGUE,
    }
    _write_json(out / "decision.json", decision)
    _write_json(
        out / "runtime.json",
        {"runtime_s": runtime_s, "wall_s": cfg.wall_s, "stages": stages, "skipped": [s for s in skipped if s], "n_new_decoders": len(primary_done)},
    )
    complete_status = "complete" if set(primary_done) >= {"R1", "R2", "R3", "R4"} else "complete_with_resource_cap"
    _write_json(out / "COMPLETE.json", {"status": complete_status, "decision": decision, "runtime_s": runtime_s, "tests": tests["all_passed"]})
    write_all_reports(out, blocked=False, tests=tests, env=env, decision=decision, cells=cells, parity=parity, skipped=[s for s in skipped if s], runtime_s=runtime_s)
    print(f"[repro] done label={primary_label} t={runtime_s:.1f}s stages={stages} skipped={skipped}", flush=True)
    return decision
