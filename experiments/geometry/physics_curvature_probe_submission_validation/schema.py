"""Typed targets. Catalog magnitude and local OOF R² must never share a label."""

from __future__ import annotations

from enum import Enum


class TargetKind(str, Enum):
    PROBE_PERFORMANCE = "probe_performance"
    CATALOG_VALUE = "catalog_value"


class ProbeTargetId(str, Enum):
    MAG_R_DESI_LOCAL_OOF_R2 = "mag_r_desi_local_oof_r2"
    MAG_R_DESI_OOF_SSE = "mag_r_desi_oof_sse"
    MAG_R_DESI_LOCAL_SST = "mag_r_desi_local_sst"
    MAG_R_DESI_LOCAL_TARGET_VAR = "mag_r_desi_local_target_var"
    MAG_R_DESI_OOF_MAE = "mag_r_desi_oof_mae"
    MAG_R_DESI_OOF_MSE = "mag_r_desi_oof_mse"
    MAG_R_DESI_OOF_NMSE = "mag_r_desi_oof_nmse"
    MAG_R_DESI_NORMALIZED_MSE = "mag_r_desi_normalized_mse"


class CatalogTargetId(str, Enum):
    MAG_R_DESI_CATALOG_VALUE = "mag_r_desi_catalog_value"


DIRECT_ERROR = (
    ProbeTargetId.MAG_R_DESI_OOF_SSE,
    ProbeTargetId.MAG_R_DESI_OOF_MAE,
    ProbeTargetId.MAG_R_DESI_OOF_MSE,
    ProbeTargetId.MAG_R_DESI_OOF_NMSE,
    ProbeTargetId.MAG_R_DESI_NORMALIZED_MSE,
)
DENOMINATOR = (
    ProbeTargetId.MAG_R_DESI_LOCAL_SST,
    ProbeTargetId.MAG_R_DESI_LOCAL_TARGET_VAR,
)
PRIMARY = ProbeTargetId.MAG_R_DESI_LOCAL_OOF_R2

AMBIGUOUS_LABELS = frozenset({"mag_r_desi", "local_r2", "y", "label", "magnitude", "mag_r"})


def kind_of(name: str) -> TargetKind:
    if name in {e.value for e in CatalogTargetId}:
        return TargetKind.CATALOG_VALUE
    if name in {e.value for e in ProbeTargetId}:
        return TargetKind.PROBE_PERFORMANCE
    raise ValueError(f"unknown target id {name!r}")


def assert_probe_performance(name: str) -> str:
    if name in AMBIGUOUS_LABELS:
        raise RuntimeError(
            f"ambiguous target {name!r}: use mag_r_desi_local_oof_r2 or mag_r_desi_catalog_value"
        )
    if kind_of(name) is TargetKind.CATALOG_VALUE:
        raise RuntimeError(f"catalog value {name} cannot be used as the primary probe-performance target")
    return name


def assert_not_catalog_vector(y: object, catalog: object, *, atol: float = 1e-8) -> None:
    import numpy as np

    a = np.asarray(y, dtype=float)
    b = np.asarray(catalog, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if int(m.sum()) < 8:
        return
    if float(np.max(np.abs(a[m] - b[m]))) <= atol:
        raise RuntimeError("primary target vector is identical to mag_r_desi_catalog_value")
    sa, sb = float(np.std(a[m])), float(np.std(b[m]))
    if sa < 1e-12 or sb < 1e-12:
        return
    pear = float(np.corrcoef(a[m], b[m])[0, 1])
    if np.isfinite(pear) and abs(pear) > 0.999 and abs(np.mean(a[m]) - np.mean(b[m])) < 0.05 * max(abs(np.mean(b[m])), 1.0) and abs(sa / sb - 1.0) < 0.05:
        raise RuntimeError(
            f"primary target is numerically interchangeable with mag_r_desi_catalog_value (pearson={pear:.6f})"
        )
