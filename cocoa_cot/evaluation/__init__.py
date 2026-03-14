"""
Evaluation module: PRR, AUROC, ECE metrics and reliability diagrams.
"""

from cocoa_cot.evaluation.metrics import prr, auroc, ece, reliability_diagram
from cocoa_cot.evaluation.quality import (
    gsm8k_accuracy,
    math500_accuracy,
    alignscore,
    livecodebench_pass_at_1,
    get_quality_fn,
)
from cocoa_cot.evaluation.calibration import compute_ece, temperature_scale

__all__ = [
    "prr",
    "auroc",
    "ece",
    "reliability_diagram",
    "gsm8k_accuracy",
    "math500_accuracy",
    "alignscore",
    "livecodebench_pass_at_1",
    "get_quality_fn",
    "compute_ece",
    "temperature_scale",
]
