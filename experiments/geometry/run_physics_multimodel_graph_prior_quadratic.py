#!/usr/bin/env python3
"""CLI: multi-model OOF probe geography + graph/quadratic dim + gated curvature."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_EXP = Path(__file__).resolve().parents[1]
if str(_EXP) not in sys.path:
    sys.path.insert(0, str(_EXP))

from geometry.physics_activation_atlas.multimodel_graph_prior_quadratic import (  # noqa: E402
    MODELS,
    MultiModelConfig,
    SCALES,
    TARGETS,
    run,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/geometry/physics_multimodel_graph_prior_quadratic",
    )
    p.add_argument(
        "--stage",
        default="all",
        help="Comma-separated stages or 'all'. "
        "Stages: prepare,global_probes,neighbourhoods,probe_geography,"
        "graph_prior,quadratic_dimension,curvature,inference,analyze",
    )
    p.add_argument(
        "--models",
        default=",".join(MODELS.keys()),
        help="Comma-separated model keys",
    )
    p.add_argument(
        "--graph-screen-models",
        default="vit_base,dinov3,clip_base",
        help="Models for Stage B graph/quadratic screen",
    )
    p.add_argument("--targets", default=",".join(TARGETS))
    p.add_argument("--scales", default=",".join(str(s) for s in SCALES))
    p.add_argument("--n-anchors", type=int, default=512)
    p.add_argument("--screen-anchors", type=int, default=96)
    p.add_argument("--probe-alpha", type=float, default=100.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--sketch-dim", type=int, default=128)
    p.add_argument("--n-sketches", type=int, default=3)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    cfg = MultiModelConfig(
        output_dir=args.output_dir,
        stage=args.stage,
        models=[m.strip() for m in args.models.split(",") if m.strip()],
        graph_screen_models=[m.strip() for m in args.graph_screen_models.split(",") if m.strip()],
        targets=[t.strip() for t in args.targets.split(",") if t.strip()],
        scales=[int(s) for s in args.scales.split(",") if s.strip()],
        n_anchors=args.n_anchors,
        screen_anchors=args.screen_anchors,
        probe_alpha=args.probe_alpha,
        seed=args.seed,
        device=args.device,
        sketch_dim=args.sketch_dim,
        n_sketches=args.n_sketches,
        force=args.force,
    )
    run(cfg)


if __name__ == "__main__":
    main()
