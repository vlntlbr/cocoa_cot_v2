"""
Calibration utilities: ECE computation and temperature scaling.

Temperature scaling fits a single scalar T on a validation set to minimise
ECE, transforming uncertainty scores u → u / T.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.optimize import minimize_scalar

from cocoa_cot.evaluation.metrics import ece

logger = logging.getLogger(__name__)


def compute_ece(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bins: int = 10,
    quality_threshold: float = 0.5,
) -> float:
    """Compute Expected Calibration Error.

    Thin wrapper around :func:`~cocoa_cot.evaluation.metrics.ece` for
    convenience.

    Args:
        uncertainty_scores: Array of uncertainty estimates.
        quality_scores: Array of quality estimates.
        n_bins: Number of confidence bins.
        quality_threshold: Threshold above which an example is "correct".

    Returns:
        ECE in [0, 1].
    """
    return ece(
        uncertainty_scores,
        quality_scores,
        n_bins=n_bins,
        quality_threshold=quality_threshold,
    )


def temperature_scale(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bins: int = 10,
    quality_threshold: float = 0.5,
    T_range: tuple[float, float] = (0.01, 100.0),
) -> tuple[float, float, float]:
    """Fit a temperature parameter T to minimise ECE on a calibration set.

    Uncertainty scores are scaled as u' = u / T.  The optimal T is found
    via scalar Brent optimisation.

    Args:
        uncertainty_scores: Uncertainty estimates (validation / holdout set).
        quality_scores: Quality estimates (validation / holdout set).
        n_bins: Number of ECE bins.
        quality_threshold: Binary correctness threshold.
        T_range: Search range for T as (T_min, T_max).

    Returns:
        Tuple of (T_opt, ece_before, ece_after) where:
            - T_opt is the fitted temperature
            - ece_before is ECE without scaling
            - ece_after is ECE with optimal scaling
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    ece_before = ece(
        uncertainty_scores, quality_scores, n_bins=n_bins, quality_threshold=quality_threshold
    )

    def _objective(T: float) -> float:
        scaled = uncertainty_scores / max(T, 1e-9)
        return ece(scaled, quality_scores, n_bins=n_bins, quality_threshold=quality_threshold)

    result = minimize_scalar(
        _objective,
        bounds=T_range,
        method="bounded",
        options={"xatol": 1e-4},
    )

    T_opt = float(result.x)
    ece_after = float(result.fun)

    logger.info(
        "Temperature scaling: T=%.4f, ECE %.4f → %.4f",
        T_opt, ece_before, ece_after,
    )
    return T_opt, ece_before, ece_after


def calibration_bins(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bins: int = 10,
    quality_threshold: float = 0.5,
) -> dict:
    """Compute per-bin calibration statistics.

    Args:
        uncertainty_scores: Uncertainty estimates.
        quality_scores: Quality estimates.
        n_bins: Number of bins.
        quality_threshold: Binary correctness threshold.

    Returns:
        Dictionary with:
            - ``"bin_confs"``: mean confidence per bin
            - ``"bin_accs"``: mean accuracy per bin
            - ``"bin_counts"``: number of examples per bin
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
    span = u_max - u_min + 1e-9
    confidence = 1.0 - (uncertainty_scores - u_min) / span
    correct = (quality_scores > quality_threshold).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_confs = []
    bin_accs = []
    bin_counts = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            bin_confs.append(np.nan)
            bin_accs.append(np.nan)
        else:
            bin_confs.append(float(confidence[mask].mean()))
            bin_accs.append(float(correct[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return {
        "bin_confs": bin_confs,
        "bin_accs": bin_accs,
        "bin_counts": bin_counts,
    }
