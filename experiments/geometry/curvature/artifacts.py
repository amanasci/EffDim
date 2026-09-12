"""Manifests, hashes, completion markers. Does not fit anything."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


COMPLETION_MARKERS = ("COMPLETE.json",)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_meta(path: Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {"path": str(p), "exists": False}
    return {"path": str(p.resolve()), "exists": True, "sha256": file_sha256(p), "bytes": int(p.stat().st_size)}


def is_complete(out_dir: Path) -> bool:
    return (Path(out_dir) / "COMPLETE.json").exists()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def decision_label(out_dir: Path) -> str | None:
    p = Path(out_dir) / "decision.json"
    if p.exists():
        d = load_json(p)
        return d.get("label") or d.get("summary_label") or d.get("primary_reproduction_label")
    complete = Path(out_dir) / "COMPLETE.json"
    if complete.exists():
        d = load_json(complete)
        return d.get("audit_interpretation") or d.get("primary_label") or d.get("label")
    return None


def source_of_truth_rank() -> tuple[str, ...]:
    return (
        "COMPLETE.json",
        "decision.json",
        "summary.json",
        "machine_readable_tables",
        "REPORT.md",
        "METHODS.md",
        "manuscript",
        "notes",
    )
