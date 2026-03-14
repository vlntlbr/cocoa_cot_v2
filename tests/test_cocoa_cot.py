"""
Tests for CoCoACoT estimator properties.

Key property checked:
  alpha=1.0 → Û = u_A * u_cons_A  (no reasoning component)
  alpha=0.0 → Û = u_R * u_cons_A  (no answer confidence)
  Û ∈ [0, 1] for valid inputs
  estimate_blackbox == u_cons_A

All tests use mocked sub-components so no GPU / model downloads are needed.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock, patch


# ── Mock factories ────────────────────────────────────────────────────────────

def _make_mock_gen_output(answer: str, chain: str, logprobs=None):
    """Create a mock GenerationOutput with the given answer and chain."""
    out = MagicMock()
    out.answer_text = answer
    out.chain_text = chain
    if logprobs is None:
        logprobs = [-0.5, -0.3, -0.2]
    out.answer_token_logprobs = logprobs
    out.token_logprobs = logprobs + [-0.1, -0.4]
    out.answer_token_entropies = [abs(lp) * 0.1 for lp in logprobs]
    return out


def _make_mock_model(answer: str = "42", chain: str = "Step 1: compute."):
    """Create a mock HFModel that always returns the same answer."""
    mock = MagicMock()
    greedy_out = _make_mock_gen_output(answer, chain)
    mock.generate_greedy.return_value = greedy_out
    mock.generate_sample.return_value = _make_mock_gen_output(answer, chain)
    return mock, greedy_out


def _make_mock_answer_sim(score: float = 1.0):
    """Cross-encoder that always returns `score`."""
    mock = MagicMock()
    mock.compute.return_value = score
    mock.compute_batch.side_effect = lambda pairs: [score] * len(pairs)
    mock.compute_one_to_many.side_effect = lambda ref, cands: [score] * len(cands)
    return mock


def _make_mock_step_sim(score: float = 0.5):
    """Step-aligned similarity that always returns `score`."""
    mock = MagicMock()
    mock.compute.return_value = score
    mock.compute_batch.side_effect = lambda ref, cands: [score] * len(cands)
    return mock


def _make_mock_parser(answer: str = "42", chain: str = "Step 1: compute."):
    mock = MagicMock()
    mock.parse.return_value = (chain, answer)
    return mock


# ── Alpha=1.0 reduces to answer-only (CoCoA) ─────────────────────────────────

class TestAlphaOne:
    def test_alpha_one_ignores_step_sim(self):
        """When alpha=1.0, u_R contribution is 0, so step_sim is effectively unused."""
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

        mock_model, greedy = _make_mock_model()
        answer_sim = _make_mock_answer_sim(score=0.8)  # u_cons_A = 1 - 0.8 = 0.2
        step_sim_high = _make_mock_step_sim(score=0.9)  # would give u_R = 0.1
        step_sim_low = _make_mock_step_sim(score=0.0)   # would give u_R = 1.0

        M = 5
        estimator_high = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim_high, alpha=1.0, M=M,
        )
        estimator_low = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim_low, alpha=1.0, M=M,
        )

        result_high = estimator_high.estimate("What is 2+2?")
        result_low = estimator_low.estimate("What is 2+2?")

        # With alpha=1.0, u_R is not used → results must be equal
        assert abs(result_high["uncertainty"] - result_low["uncertainty"]) < 1e-6, (
            f"alpha=1.0 should ignore step_sim: {result_high['uncertainty']} vs {result_low['uncertainty']}"
        )


class TestAlphaZero:
    def test_alpha_zero_ignores_answer_confidence(self):
        """When alpha=0.0, u_A contribution is 0."""
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT
        from cocoa_cot.uncertainty.information import PPLEstimator

        mock_model, greedy = _make_mock_model()
        answer_sim = _make_mock_answer_sim(score=0.6)
        step_sim = _make_mock_step_sim(score=0.7)

        estimator = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim, alpha=0.0, M=5,
        )
        result = estimator.estimate("What is 2+2?")

        # u_A component is 0 → Û = u_R * u_cons_A
        u_R = result["u_R"]
        u_cons = result["u_cons_A"]
        expected = u_R * u_cons
        assert abs(result["uncertainty"] - expected) < 1e-6


# ── Uncertainty always in [0, 1] ──────────────────────────────────────────────

class TestUncertaintyBounds:
    @pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 0.75, 1.0])
    def test_uncertainty_in_unit_interval(self, alpha):
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

        mock_model, _ = _make_mock_model()
        answer_sim = _make_mock_answer_sim(score=0.4)
        step_sim = _make_mock_step_sim(score=0.6)

        estimator = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim, alpha=alpha, M=5,
        )
        result = estimator.estimate("Test prompt?")
        u = result["uncertainty"]
        assert 0.0 <= u <= 1.0 + 1e-6, f"alpha={alpha}: uncertainty={u} out of [0,1]"


# ── estimate_blackbox == u_cons_A ─────────────────────────────────────────────

class TestBlackboxEstimate:
    def test_blackbox_equals_cons(self):
        """estimate_blackbox must equal Û_cons_A (pure answer consistency)."""
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

        mock_model, _ = _make_mock_model()
        answer_sim = _make_mock_answer_sim(score=0.7)
        step_sim = _make_mock_step_sim(score=0.5)

        estimator = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim, alpha=0.5, M=5,
        )

        greedy_answer = "42"
        sampled_answers = ["42", "42", "41", "42", "43"]
        bb = estimator.estimate_blackbox(greedy_answer, sampled_answers)

        # With score=0.7 for all pairs, u_cons_A = 1 - 0.7 = 0.3
        assert 0.0 <= bb <= 1.0
        # Score=0.7 with 5 samples → cons = 1 - mean([0.7]*5) = 0.3
        assert abs(bb - 0.3) < 1e-5, f"Expected 0.3, got {bb}"


# ── Result dict has required keys ────────────────────────────────────────────

class TestResultKeys:
    def test_estimate_returns_all_keys(self):
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

        mock_model, _ = _make_mock_model()
        answer_sim = _make_mock_answer_sim()
        step_sim = _make_mock_step_sim()

        estimator = CoCoACoT(
            model=mock_model, answer_similarity=answer_sim,
            step_similarity=step_sim, alpha=0.5, M=3,
        )
        result = estimator.estimate("Prompt?")

        required_keys = {
            "uncertainty", "u_A", "u_R", "u_cons_A",
            "greedy_answer", "greedy_chain",
            "sampled_answers", "sampled_chains", "alpha",
        }
        for key in required_keys:
            assert key in result, f"Missing key: {key}"

    def test_alpha_echoed_correctly(self):
        from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

        mock_model, _ = _make_mock_model()
        estimator = CoCoACoT(
            model=mock_model,
            answer_similarity=_make_mock_answer_sim(),
            step_similarity=_make_mock_step_sim(),
            alpha=0.37, M=3,
        )
        result = estimator.estimate("Prompt?")
        assert abs(result["alpha"] - 0.37) < 1e-9
