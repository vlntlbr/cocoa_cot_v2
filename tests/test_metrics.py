"""
Tests for evaluation metrics: PRR, AUROC, ECE.
All tests are offline — no model required.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from cocoa_cot.evaluation.metrics import prr, auroc, ece


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture()
def perfect_oracle():
    """Uncertainty exactly identifies wrong answers (uncertainty=1 ↔ quality=0)."""
    quality = np.array([1, 1, 1, 0, 0, 0, 1, 0, 1, 0], dtype=float)
    # Perfect oracle: uncertainty high when quality low
    uncertainty = 1.0 - quality
    return uncertainty, quality


@pytest.fixture()
def random_scores(rng=None):
    """Uniformly random uncertainty scores."""
    rng = np.random.default_rng(42)
    quality = rng.integers(0, 2, size=100).astype(float)
    uncertainty = rng.uniform(0, 1, size=100)
    return uncertainty, quality


@pytest.fixture()
def constant_scores():
    """All uncertainty scores the same."""
    quality = np.array([1, 0, 1, 0, 1], dtype=float)
    uncertainty = np.ones(5) * 0.5
    return uncertainty, quality


# ── PRR tests ─────────────────────────────────────────────────────────────────

class TestPRR:
    def test_perfect_oracle_near_one(self, perfect_oracle):
        unc, qual = perfect_oracle
        score = prr(unc, qual, rejection_max=0.5)
        assert score > 0.8, f"Expected PRR > 0.8 for oracle, got {score}"

    def test_random_near_zero(self, random_scores):
        unc, qual = random_scores
        score = prr(unc, qual, rejection_max=0.5)
        # Random should be close to 0 (not necessarily negative due to discretization)
        assert score < 0.3, f"Expected PRR < 0.3 for random, got {score}"

    def test_constant_uncertainty_returns_finite(self, constant_scores):
        unc, qual = constant_scores
        score = prr(unc, qual, rejection_max=0.5)
        assert math.isfinite(score)

    def test_output_in_range(self, perfect_oracle):
        unc, qual = perfect_oracle
        score = prr(unc, qual, rejection_max=0.5)
        assert -1.1 <= score <= 1.1  # PRR is normalized; can be slightly outside [-1,1] due to edge cases

    def test_different_rejection_max(self, perfect_oracle):
        unc, qual = perfect_oracle
        s1 = prr(unc, qual, rejection_max=0.3)
        s2 = prr(unc, qual, rejection_max=0.8)
        # Both should be finite
        assert math.isfinite(s1) and math.isfinite(s2)

    def test_all_correct_returns_finite(self):
        unc = np.linspace(0, 1, 50)
        qual = np.ones(50)
        score = prr(unc, qual)
        assert math.isfinite(score)

    def test_all_wrong_returns_finite(self):
        unc = np.linspace(0, 1, 50)
        qual = np.zeros(50)
        score = prr(unc, qual)
        assert math.isfinite(score)


# ── AUROC tests ───────────────────────────────────────────────────────────────

class TestAUROC:
    def test_perfect_oracle_near_one(self, perfect_oracle):
        unc, qual = perfect_oracle
        score = auroc(unc, qual)
        assert score > 0.9, f"Expected AUROC > 0.9 for oracle, got {score}"

    def test_random_near_half(self, random_scores):
        unc, qual = random_scores
        score = auroc(unc, qual)
        assert 0.3 < score < 0.7, f"Expected AUROC ≈ 0.5 for random, got {score}"

    def test_output_in_unit_interval(self, perfect_oracle):
        unc, qual = perfect_oracle
        score = auroc(unc, qual)
        assert 0.0 <= score <= 1.0

    def test_worst_oracle(self, perfect_oracle):
        """Inverted scores → AUROC near 0."""
        unc, qual = perfect_oracle
        # Invert: high uncertainty when quality is high
        inverted_unc = 1.0 - unc
        score = auroc(inverted_unc, qual)
        assert score < 0.2, f"Expected AUROC < 0.2 for inverted oracle, got {score}"

    def test_all_same_class_returns_finite(self):
        unc = np.linspace(0, 1, 10)
        qual = np.ones(10)  # only one class
        score = auroc(unc, qual)
        assert math.isfinite(score)


# ── ECE tests ─────────────────────────────────────────────────────────────────

class TestECE:
    def test_perfectly_calibrated_near_zero(self):
        """If predicted confidence equals empirical accuracy, ECE ≈ 0."""
        n = 100
        rng = np.random.default_rng(7)
        quality = rng.uniform(0, 1, n)
        # uncertainty = 1 - quality (confidence = quality)
        uncertainty = 1.0 - quality
        score = ece(uncertainty, quality, n_bins=10)
        assert score < 0.2, f"Expected low ECE for calibrated scores, got {score}"

    def test_random_scores_positive(self, random_scores):
        unc, qual = random_scores
        score = ece(unc, qual, n_bins=10)
        assert score >= 0.0

    def test_ece_in_unit_interval(self, random_scores):
        unc, qual = random_scores
        score = ece(unc, qual, n_bins=10)
        assert 0.0 <= score <= 1.0

    def test_n_bins_parameter(self, random_scores):
        unc, qual = random_scores
        s5 = ece(unc, qual, n_bins=5)
        s20 = ece(unc, qual, n_bins=20)
        # Both should be valid
        assert math.isfinite(s5) and math.isfinite(s20)

    def test_constant_uncertainty(self, constant_scores):
        unc, qual = constant_scores
        score = ece(unc, qual, n_bins=10)
        assert math.isfinite(score)


# ── Reliability diagram (smoke test) ─────────────────────────────────────────

class TestReliabilityDiagram:
    def test_no_crash(self, tmp_path):
        from cocoa_cot.evaluation.metrics import reliability_diagram
        unc = np.linspace(0, 1, 100)
        qual = (unc < 0.5).astype(float)
        out = tmp_path / "rel_diag.png"
        reliability_diagram(unc, qual, n_bins=10, save_path=str(out))
        # File may not exist if matplotlib unavailable, just check no exception
