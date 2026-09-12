"""Typed names. Catalogue labels, probe performance and curvature are not interchangeable."""

from __future__ import annotations

from enum import Enum


class Estimator(str, Enum):
    Q_KH_CROSS = "KHcross"
    Q_KDIR_CROSS = "Kdircross"
    D_FULL = "D_full"
    D_NORMALIZED_FULL = "D_normalized_full"
    D_RESIDUAL = "D_residual"
    E_Q_TASK_ALIGNED = "E_Q_task_aligned"
    E_D_TASK_ALIGNED = "E_D_task_aligned"
    M_DELTA = "M_delta"
    A_FULL = "A_full"


class OutcomeKind(str, Enum):
    CATALOGUE_LABEL = "catalogue_label"
    GLOBAL_R2 = "global_r2"
    GLOBAL_MSE = "global_mse"
    PATCH_R2 = "patch_r2"
    PATCH_MSE = "patch_mse"
    ADAPTATION_GAIN = "adaptation_gain"


class TargetId(str, Enum):
    MAG_R_DESI = "mag_r_desi"
    PHOTO_Z = "photo_z"
    SMOOTH_FRACTION = "smooth_fraction"
    STELLAR_MASS = "stellar_mass"


def assert_not_catalogue_as_performance(kind: OutcomeKind) -> None:
    if kind is OutcomeKind.CATALOGUE_LABEL:
        raise TypeError("catalogue label is not a probe-performance outcome")
