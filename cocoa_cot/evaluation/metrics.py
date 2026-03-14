"""
Evaluation metrics: PRR, AUROC, ECE, and reliability diagrams.

Reference equations:
    PRR = (AUC_unc - AUC_rnd) / (AUC_oracle - AUC_rnd)    [Eq. 22]
    AUROC: standard scikit-learn roc_auc_score
    ECE = Σ_b (|b|/n) |acc(b) - conf(b)|
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from scipy.integrate import trapezoid
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


def prr(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    rejection_max: float = 0.5,
    n_points: int = 100,
) -> float:
    """Prediction Rejection Ratio.

    Measures how well uncertainty scores identify low-quality examples by
    comparing the area under the rejection curve to oracle and random baselines.

    For each rejection threshold τ ∈ [0, rejection_max]:
        - Reject examples with uncertainty_score > quantile(τ)
        - Compute mean quality of remaining examples

    PRR = (AUC_unc - AUC_rnd) / (AUC_oracle - AUC_rnd)

    Args:
        uncertainty_scores: Array of uncertainty estimates.  Higher = more
            uncertain = should be rejected first.
        quality_scores: Array of quality estimates.  Higher = better quality.
        rejection_max: Maximum rejection fraction in [0, 1].
        n_points: Number of rejection thresholds to evaluate.

    Returns:
        PRR in approximately [-1, 1].  Higher is better.  A perfect oracle
        scorer returns ~1.0; a random scorer returns ~0.0.
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    n = len(uncertainty_scores)
    if n == 0:
        return 0.0

    thresholds = np.linspace(0.0, rejection_max, n_points)

    def _rejection_curve(scores_to_reject_first: np.ndarray) -> np.ndarray:
        """Compute rejection curve for a given ordering."""
        # Argsort ascending: lowest score → kept last
        order = np.argsort(scores_to_reject_first)[::-1]  # descending
        curve = np.zeros(len(thresholds))
        for i, tau in enumerate(thresholds):
            n_reject = int(np.ceil(tau * n))
            kept_indices = order[n_reject:]
            if len(kept_indices) == 0:
                curve[i] = 0.0
            else:
                curve[i] = quality_scores[kept_indices].mean()
        return curve

    # AUC under uncertainty-based rejection (reject highest uncertainty first)
    curve_unc = _rejection_curve(uncertainty_scores)
    auc_unc = float(trapezoid(curve_unc, thresholds))

    # AUC under random rejection (expected quality = constant mean)
    mean_quality = float(quality_scores.mean())
    auc_rnd = float(mean_quality * rejection_max)

    # AUC under oracle rejection (reject lowest-quality first)
    # Oracle: sort by quality ascending → reject worst first
    oracle_sort = np.argsort(quality_scores)  # ascending: worst first
    order_oracle = oracle_sort[::-1]  # now: best first, but we want to
    # actually: oracle rejects worst first → keep best
    # i.e. sort ascending, first rejected are worst
    def _oracle_curve() -> np.ndarray:
        order = np.argsort(quality_scores)  # ascending: worst first → reject first
        curve = np.zeros(len(thresholds))
        for i, tau in enumerate(thresholds):
            n_reject = int(np.ceil(tau * n))
            kept_indices = order[n_reject:]  # keep the best ones
            if len(kept_indices) == 0:
                curve[i] = 0.0
            else:
                curve[i] = quality_scores[kept_indices].mean()
        return curve

    curve_oracle = _oracle_curve()
    auc_oracle = float(trapezoid(curve_oracle, thresholds))

    denom = auc_oracle - auc_rnd
    if abs(denom) < 1e-10:
        return 0.0

    return float((auc_unc - auc_rnd) / denom)


def auroc(
    uncertainty_scores: np.ndarray,
    correct: np.ndarray,
) -> float:
    """Area Under the ROC Curve for binary correctness prediction.

    Uses uncertainty as the score for predicting *incorrect* (low quality)
    examples, i.e. high uncertainty → predicted incorrect.

    Args:
        uncertainty_scores: Higher = more uncertain.
        correct: Binary array.  1 = correct (high quality), 0 = incorrect.

    Returns:
        AUROC in [0, 1].  Higher is better.
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.int32)

    if len(np.unique(correct)) < 2:
        logger.warning("auroc: only one class present; returning 0.5")
        return 0.5

    # Uncertainty predicts *incorrectness* (1 - correct)
    # → higher uncertainty → predicted class 0 (incorrect)
    try:
        return float(roc_auc_score(correct, -uncertainty_scores))
    except Exception as exc:
        logger.warning("auroc computation failed: %s", exc)
        return 0.5


def ece(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bins: int = 10,
    quality_threshold: float = 0.5,
) -> float:
    """Expected Calibration Error.

    Converts uncertainty scores to a ``confidence`` proxy via:
        confidence = 1 - (uncertainty - min) / (max - min + eps)

    Bins examples by confidence and computes:
        ECE = Σ_b (|b| / n) |acc(b) - conf(b)|

    where acc(b) = fraction of examples in bin b with quality > quality_threshold.

    Args:
        uncertainty_scores: Array of uncertainty estimates.
        quality_scores: Array of quality estimates.
        n_bins: Number of confidence bins.
        quality_threshold: Threshold above which an example is "correct".

    Returns:
        ECE in [0, 1].  Lower is better.
    """
    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    n = len(uncertainty_scores)
    if n == 0:
        return 0.0

    # Normalise to [0, 1] confidence
    u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
    span = u_max - u_min + 1e-9
    confidence = 1.0 - (uncertainty_scores - u_min) / span  # high conf = low uncertainty

    correct = (quality_scores > quality_threshold).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    ece_val = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidence[mask].mean()
        bin_acc = correct[mask].mean()
        ece_val += (mask.sum() / n) * abs(bin_acc - bin_conf)

    return float(ece_val)


def reliability_diagram(
    uncertainty_scores: np.ndarray,
    quality_scores: np.ndarray,
    n_bins: int = 10,
    quality_threshold: float = 0.5,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot a reliability (calibration) diagram.

    Shows confidence (x-axis) vs. accuracy (y-axis) with a perfect
    calibration diagonal for reference.

    Args:
        uncertainty_scores: Array of uncertainty estimates.
        quality_scores: Array of quality estimates.
        n_bins: Number of bins.
        quality_threshold: Threshold above which an example is correct.
        save_path: If provided, saves the figure to this path.
        title: Plot title.
    """
    import matplotlib.pyplot as plt

    uncertainty_scores = np.asarray(uncertainty_scores, dtype=np.float64)
    quality_scores = np.asarray(quality_scores, dtype=np.float64)

    u_min, u_max = uncertainty_scores.min(), uncertainty_scores.max()
    span = u_max - u_min + 1e-9
    confidence = 1.0 - (uncertainty_scores - u_min) / span
    correct = (quality_scores > quality_threshold).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    bin_accs = []
    bin_confs = []
    bin_counts = []

    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        mask = (confidence >= lo) & (confidence < hi)
        if b == n_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        if mask.sum() == 0:
            bin_accs.append(np.nan)
            bin_confs.append(bin_centers[b])
        else:
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confidence[mask].mean())
        bin_counts.append(int(mask.sum()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: reliability diagram
    ax = axes[0]
    valid = ~np.isnan(bin_accs)
    ax.bar(
        np.array(bin_confs)[valid],
        np.array(bin_accs)[valid],
        width=1.0 / n_bins,
        alpha=0.7,
        color="steelblue",
        label="Model",
    )
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Right: confidence histogram
    ax2 = axes[1]
    ax2.bar(bin_centers, bin_counts, width=1.0 / n_bins, alpha=0.7, color="orange")
    ax2.set_xlabel("Confidence")
    ax2.set_ylabel("Count")
    ax2.set_title("Confidence Distribution")

    ece_val = ece(uncertainty_scores, quality_scores, n_bins=n_bins, quality_threshold=quality_threshold)
    fig.suptitle(f"{title} (ECE = {ece_val:.4f})", fontsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Reliability diagram saved to %s", save_path)
    else:
        plt.show()

    plt.close(fig)
