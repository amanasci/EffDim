#!/usr/bin/env python3
"""CLI: per-patch quadratic flattening vs tangent / ambient cross-model mKNN.

Smoke outputs → outputs/geometry/quadratic_mknn/smoke/

Does not overwrite curvature atlas or retrieval-information outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.paths import platonic_root  # noqa: E402
from geometry.physics_activation_atlas.quadratic_mknn import (  # noqa: E402
    QuadraticMKNNConfig,
    evaluate_smoke,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/quadratic_mknn/smoke",
        help="Under PLATONIC_ROOT (default: smoke)",
    )
    p.add_argument(
        "--models",
        default="vit_base,dinov3,clip_base",
        help="Comma-separated Physics model keys",
    )
    p.add_argument(
        "--space",
        choices=("dense", "sae", "sae_idf"),
        default="dense",
        help="Chart/distance representation: dense activations, TopK SAE codes, or SAE×IDF",
    )
    p.add_argument(
        "--sae-tag",
        default="",
        help="SAE checkpoint tag (default: first of F2048_k20/k22/…)",
    )
    p.add_argument(
        "--chart-scales",
        default="256,512,1024,2048",
        help="K: neighbours used to fit each local chart",
    )
    p.add_argument(
        "--retrieval-ks",
        default="5,10,20,50",
        help="k: final mKNN neighbourhood sizes",
    )
    p.add_argument("--n-anchors", type=int, default=96)
    p.add_argument("--dim-screen-anchors", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--candidate-multipliers",
        default="4,10,20",
        help="Pool size multipliers relative to k (diagnostics)",
    )
    p.add_argument("--candidate-pool-min", type=int, default=512)
    p.add_argument(
        "--primary-multiplier",
        type=int,
        default=20,
        help="Ambient candidate pool = max(min, multiplier * max_k)",
    )
    p.add_argument(
        "--patch-mode",
        choices=("model_specific", "shared"),
        default="model_specific",
    )
    p.add_argument(
        "--shared-reference-model",
        default="vit_base",
        help="Reference model for shared patch neighbourhoods",
    )
    p.add_argument(
        "--phase2",
        action="store_true",
        help="Also run Q_T/Q_R/B^S ablations, geodesic, random-B^S null",
    )
    p.add_argument("--invert-lambda", type=float, default=1e-2)
    p.add_argument("--invert-iters", type=int, default=8)
    p.add_argument("--platonic-root", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    cfg = QuadraticMKNNConfig(
        output_dir=args.output_dir,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        space=args.space,
        sae_tag=args.sae_tag,
        chart_scales_K=[int(x) for x in args.chart_scales.split(",") if x.strip()],
        retrieval_ks=[int(x) for x in args.retrieval_ks.split(",") if x.strip()],
        n_anchors=args.n_anchors,
        dim_screen_anchors=args.dim_screen_anchors,
        seed=args.seed,
        device=args.device,
        candidate_multipliers=[
            int(x) for x in args.candidate_multipliers.split(",") if x.strip()
        ],
        candidate_pool_min=args.candidate_pool_min,
        primary_multiplier=args.primary_multiplier,
        patch_mode=args.patch_mode,
        shared_reference_model=args.shared_reference_model,
        phase2=args.phase2,
        invert_lambda=args.invert_lambda,
        invert_iters=args.invert_iters,
        force=args.force,
    )
    root = platonic_root(args.platonic_root)
    evaluate_smoke(root, cfg)


if __name__ == "__main__":
    main()
