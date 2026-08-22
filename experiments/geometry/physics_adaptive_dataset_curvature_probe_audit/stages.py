"""Orchestrate the audit. Never writes into completed output trees."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from .classify import audit_label, classify_root_causes
from .config import DEFAULT_THRESHOLDS, PRESERVED
from .controls import run_controls
from .desi_alignment import run_desi_alignment
from .inference import run_global, run_transitions
from .inventory import build_inventory
from .parity import run_parity
from .pipeline import AuditConfig, assert_not_preserved, sources_available, write_json
from .plots import write_figures
from .reliability import run_reliability
from .report import write_audit_complete, write_methods, write_report
from .sample_sizes import run_sample_sizes


def run(cfg: AuditConfig) -> dict[str, Any]:
    t0 = time.time()
    from .pipeline import platonic_root

    root = platonic_root()
    out = cfg.resolved(root)
    assert_not_preserved(out, root)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("cache", "figures", "logs"):
        (out / sub).mkdir(exist_ok=True)
    if not sources_available(root, cfg):
        raise RuntimeError("completed discovery/adaptive trees are not visible from platonic_root(); refuse to invent them")

    ctx: dict[str, Any] = {"thresholds": DEFAULT_THRESHOLDS, "t0": t0}
    if cfg.stage in ("all", "inventory"):
        ctx["inv"] = build_inventory(root, cfg)
        print(f"[adcp-audit] inventory shared={len(ctx['inv']['shared_sids'])}", flush=True)
    if cfg.stage in ("all", "parity"):
        ctx["parity"] = run_parity(root, cfg, ctx["inv"])
        print(f"[adcp-audit] KH identical={ctx['parity']['kh_identical']} probe_mismatch={ctx['parity']['probe_quantity_mismatch']}", flush=True)
    if cfg.stage in ("all", "controls"):
        ctx["controls"] = run_controls(root, cfg, ctx["inv"], ctx["parity"])
        print(f"[adcp-audit] delta_frozen={ctx['controls']['delta_frozen_ctl']:.6f}", flush=True)
    if cfg.stage in ("all", "desi"):
        ctx["desi"] = run_desi_alignment(root, cfg)
        print(f"[adcp-audit] DESI {ctx['desi']['status']}", flush=True)
    if cfg.stage in ("all", "sample_sizes"):
        ctx["sizes"] = run_sample_sizes(root, cfg, ctx["inv"], ctx["desi"])
        print(f"[adcp-audit] sample-size rows={len(ctx['sizes'])}", flush=True)
    if cfg.stage in ("all", "inference"):
        print("[adcp-audit] global scientific family (no DESI) ...", flush=True)
        ctx["global_scientific"] = run_global(root, cfg, ctx["inv"], include_desi=False, tag="scientific")
        print(
            f"[adcp-audit] WY p={ctx['global_scientific']['wy']['p_report']} "
            f"maxT p={ctx['global_scientific']['maxT']['p_report']}",
            flush=True,
        )
        if not cfg.smoke:
            print("[adcp-audit] global as-published family (DESI unaligned, not scientific) ...", flush=True)
            ctx["global_published"] = run_global(root, cfg, ctx["inv"], include_desi=True, tag="as_published_unaligned")
        else:
            ctx["global_published"] = ctx["global_scientific"]
        ctx["global_scientific"]["wy_table"].to_csv(out / "global_minp_results.csv", index=False)
        ctx["global_scientific"]["maxT_table"].to_csv(out / "global_studentized_maxT_results.csv", index=False)
        if ctx.get("global_published") is not None and ctx["global_published"] is not ctx["global_scientific"]:
            ctx["global_published"]["wy_table"].to_csv(out / "cache" / "global_minp_as_published_unaligned.csv", index=False)
        ctx["transitions"] = run_transitions(root, cfg, ctx["controls"], ctx["sizes"])
    if cfg.stage in ("all", "reliability"):
        ctx["reliability"] = run_reliability(root, cfg)
    if cfg.stage in ("all", "analyze", "report"):
        ctx["causes"] = classify_root_causes(ctx["parity"], ctx["desi"], ctx["controls"])
        ctx["audit_label"] = audit_label(ctx["causes"], ctx["parity"], ctx["desi"])
        write_json(
            out / "root_cause.json",
            {
                "causes": ctx["causes"],
                "audit_label": ctx["audit_label"],
                "first_divergence": "probe_quantity: local_r2 vs catalog mag_r_desi",
                "kh_identical_at_12_16_20": ctx["parity"].get("kh_identical"),
                "anchor_set_equal": ctx["parity"]["anchors"].get("set_equal"),
                "desi": ctx["desi"].get("status"),
                "repair": "reuse per-anchor K_H; restore discovery curve with local_r2; exclude DESI labels from scientific conclusions; no geometry refit",
                "sixteen_anchor_curvature_test": "not required; K_H already identical",
                "scale": "deferred",
            },
            force=cfg.force,
        )
        ctx["runtime_s"] = time.time() - t0
        ctx["n_tests"] = 18
        write_figures(out)
        write_methods(out, cfg, ctx)
        write_report(out, cfg, ctx)
        write_audit_complete(out, cfg, ctx)
        # refuse scientific COMPLETE.json
        if (out / "COMPLETE.json").exists():
            raise RuntimeError("scientific COMPLETE.json must not be written by this audit")
        print(f"[adcp-audit] done in {ctx['runtime_s']:.1f}s label={ctx['audit_label']}", flush=True)
    return ctx
