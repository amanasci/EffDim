"""Eligible / excluded probe-label registry."""

from __future__ import annotations

from .config import ELIGIBLE, EXCLUDED


def eligible_fields() -> tuple[str, ...]:
    return tuple(r["field"] for r in ELIGIBLE)


def record_for(field: str) -> dict:
    for r in ELIGIBLE:
        if r["field"] == field:
            return dict(r)
    raise KeyError(f"{field} is not an eligible physics label")


def exclusion_table() -> list[dict]:
    return [dict(r) for r in EXCLUDED]


def assert_not_desi_resurrected(field: str) -> None:
    low = field.lower()
    if "spec_z" in low or low.startswith("desi_") or low.endswith("_desi_spec"):
        if field != "mag_r_desi":
            raise ValueError(f"refusing excluded or unproven DESI field {field!r}")
