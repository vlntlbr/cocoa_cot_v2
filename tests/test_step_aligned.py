"""
Tests for StepAlignedSimilarity (Eq. 11).

Uses a mock cross-encoder similarity so no GPU / model downloads are needed.
"""

from __future__ import annotations

import math
import pytest
import numpy as np

from unittest.mock import MagicMock, patch


# ── Helper: mock CrossEncoderSimilarity ────────────────────────────────────────

def _make_mock_ce(score_fn=None):
    """
    Return a mock CrossEncoderSimilarity that uses score_fn(text_a, text_b) -> float.
    Defaults to exact-match similarity (1.0 if equal else 0.0).
    """
    if score_fn is None:
        def score_fn(a, b):
            return 1.0 if a.strip() == b.strip() else 0.0

    mock = MagicMock()

    def _compute_batch(pairs):
        return [score_fn(a, b) for a, b in pairs]

    mock.compute_batch.side_effect = _compute_batch
    return mock


# ── Import StepAlignedSimilarity ──────────────────────────────────────────────

@pytest.fixture()
def step_sim():
    from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity
    ce = _make_mock_ce()
    return StepAlignedSimilarity(sentence_similarity=ce)


# ── Identical chains → 1.0 ───────────────────────────────────────────────────

class TestIdenticalChains:
    def test_single_step_chains(self, step_sim):
        chain = "The answer is 42."
        score = step_sim.compute(chain, chain)
        assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"

    def test_multi_step_chains(self, step_sim):
        chain = "Step 1: A\nStep 2: B\nStep 3: C"
        score = step_sim.compute(chain, chain)
        assert abs(score - 1.0) < 1e-6, f"Expected 1.0, got {score}"


# ── Completely different chains → ~0.0 ───────────────────────────────────────

class TestOrthogonalChains:
    def test_no_overlapping_steps(self, step_sim):
        chain_a = "Step 1: foo\nStep 2: bar"
        chain_b = "Step 1: baz\nStep 2: qux"
        score = step_sim.compute(chain_a, chain_b)
        # With exact-match mock, all cross-scores should be 0
        assert score < 1e-6, f"Expected ~0.0, got {score}"


# ── Asymmetry property ────────────────────────────────────────────────────────

class TestAsymmetry:
    def test_asymmetric_score_is_not_commutative(self):
        """
        s_step(A,B) is computed as mean-over-A-steps of max-over-B-steps.
        With reference A having 1 step matched and B having 3 steps,
        the direction matters.
        """
        from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity

        # ce scores: step A1 matches B1=1.0, A1 vs B2=0.0, A1 vs B3=0.0
        # s_step(A,B) = max(1.0, 0.0, 0.0) = 1.0
        # s_step(B,A): B1 vs A1=1.0 → max=1.0; B2 vs A1=0.0; B3 vs A1=0.0
        # s_step(B,A) = mean(1.0, 0.0, 0.0) = 0.333
        call_counter = {"n": 0}

        def score_fn(a: str, b: str) -> float:
            # Only "step_A1" matches "step_B1"
            if a.strip() == "step_A1" and b.strip() == "step_B1":
                return 1.0
            if a.strip() == "step_B1" and b.strip() == "step_A1":
                return 1.0
            return 0.0

        ce = _make_mock_ce(score_fn)
        sim = StepAlignedSimilarity(sentence_similarity=ce)

        chain_A = "step_A1"  # 1 step
        chain_B = "step_B1\nstep_B2\nstep_B3"  # 3 steps

        s_A_B = sim.compute(chain_A, chain_B)  # reference=A (1 step)
        s_B_A = sim.compute(chain_B, chain_A)  # reference=B (3 steps)

        assert abs(s_A_B - 1.0) < 1e-6, f"s(A,B) should be 1.0, got {s_A_B}"
        assert abs(s_B_A - (1.0 / 3.0)) < 1e-5, f"s(B,A) should be 1/3, got {s_B_A}"


# ── Batch compute ─────────────────────────────────────────────────────────────

class TestBatchCompute:
    def test_batch_matches_individual(self, step_sim):
        from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity

        # Rebuild with same mock for determinism
        ce = _make_mock_ce()
        sim = StepAlignedSimilarity(sentence_similarity=ce)

        ref = "Step 1: A\nStep 2: B"
        candidates = [
            "Step 1: A\nStep 2: B",   # identical
            "Step 1: X\nStep 2: Y",   # different
            "Step 1: A\nStep 2: Z",   # partial
        ]

        batch_scores = sim.compute_batch(ref, candidates)
        individual_scores = [sim.compute(ref, c) for c in candidates]

        for i, (b, ind) in enumerate(zip(batch_scores, individual_scores)):
            assert abs(b - ind) < 1e-5, f"Mismatch at index {i}: batch={b}, individual={ind}"

    def test_batch_returns_correct_length(self, step_sim):
        from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity

        ce = _make_mock_ce()
        sim = StepAlignedSimilarity(sentence_similarity=ce)
        ref = "A single step."
        candidates = ["step one", "step two", "step three", "step four"]
        batch_scores = sim.compute_batch(ref, candidates)
        assert len(batch_scores) == 4


# ── Empty chain handling ──────────────────────────────────────────────────────

class TestEmptyChain:
    def test_empty_reference_does_not_crash(self, step_sim):
        score = step_sim.compute("", "Step 1: something")
        assert 0.0 <= score <= 1.0

    def test_empty_candidate_does_not_crash(self, step_sim):
        score = step_sim.compute("Step 1: something", "")
        assert 0.0 <= score <= 1.0

    def test_both_empty_does_not_crash(self, step_sim):
        score = step_sim.compute("", "")
        assert 0.0 <= score <= 1.0


# ── Score clamped to [0, 1] ───────────────────────────────────────────────────

class TestScoreBounds:
    def test_score_in_unit_interval(self):
        from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity

        def wild_score_fn(a, b):
            # Simulate a model that sometimes returns out-of-range values
            return 1.5

        ce = _make_mock_ce(wild_score_fn)
        sim = StepAlignedSimilarity(sentence_similarity=ce)

        chain_a = "Step 1: A\nStep 2: B"
        chain_b = "Step 1: C\nStep 2: D"
        score = sim.compute(chain_a, chain_b)
        assert 0.0 <= score <= 1.0 + 1e-6  # may be clamped or averaged
