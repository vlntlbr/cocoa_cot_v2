"""
CoCoA-CoT: the proposed uncertainty quantification method.

Implements Eq. 14 from the paper:

    Û_CoCoA-CoT(y* | x) = [α · u_A(a*|c*,x) + (1-α) · u_R(c*|x)] · Û_cons_A(a*|x)

Components:
    u_A  — answer-level confidence (MSP, PPL, or MTE over answer tokens)
    u_R  — reasoning coherence uncertainty (step-aligned similarity dissimilarity)
    Û_cons_A — answer semantic consistency (standard MBR consistency on answers)

The estimator also supports the black-box variant:
    Û_BB = Û_cons_A

and validation property:
    α=1 reduces to CoCoA applied to answer spans.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from cocoa_cot.models.base import BaseModel, GenerationOutput
from cocoa_cot.parsing.chain_parser import ChainParser
from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity
from cocoa_cot.uncertainty.information import MSPEstimator, PPLEstimator, MTEEstimator

logger = logging.getLogger(__name__)

_CONF_ESTIMATORS = {
    "msp": MSPEstimator,
    "ppl": PPLEstimator,
    "mte": MTEEstimator,
}

_EPS = 1e-6  # numerical guard for similarity clamping


class CoCoACoT:
    """CoCoA-CoT uncertainty quantification.

    Û_CoCoA-CoT(y* | x) = [α · u_A + (1-α) · u_R] · Û_cons_A

    Args:
        model: :class:`~cocoa_cot.models.BaseModel` wrapper.
        answer_similarity: :class:`~cocoa_cot.similarity.BaseSimilarity` for
            computing answer-level semantic consistency ``Û_cons_A``.
        step_similarity: :class:`~cocoa_cot.similarity.StepAlignedSimilarity`
            for computing reasoning coherence ``u_R``.
        parser: :class:`~cocoa_cot.parsing.ChainParser` instance.
        alpha: Mixing parameter α ∈ [0, 1].  α=1 → answer-only (reduces to
            CoCoA);  α=0 → reasoning-only.
        M: Default number of stochastic samples per prompt.
        confidence_type: One of ``"msp"``, ``"ppl"``, or ``"mte"``.
        temperature: Sampling temperature for the M samples.
        top_k: Top-k parameter.
        top_p: Nucleus parameter.
    """

    def __init__(
        self,
        model: BaseModel,
        answer_similarity: BaseSimilarity,
        step_similarity: StepAlignedSimilarity,
        parser: ChainParser,
        alpha: float = 0.5,
        M: int = 10,
        confidence_type: str = "ppl",
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if confidence_type not in _CONF_ESTIMATORS:
            raise ValueError(
                f"confidence_type must be one of {list(_CONF_ESTIMATORS)}, "
                f"got {confidence_type!r}"
            )

        self.model = model
        self.answer_sim = answer_similarity
        self.step_sim = step_similarity
        self.parser = parser
        self.alpha = alpha
        self.M = M
        self.confidence_type = confidence_type
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p

        # CoT-mode confidence estimator (uses answer tokens only)
        self._conf_estimator = _CONF_ESTIMATORS[confidence_type](cot_mode=True)

    # ── Public estimation API ─────────────────────────────────────────────────

    def estimate(self, prompt: str, M: Optional[int] = None) -> dict:
        """Full CoCoA-CoT UQ estimation for a single prompt.

        Args:
            prompt: Input prompt string.
            M: Number of samples (overrides default if provided).

        Returns:
            Dictionary with keys:

            - ``"uncertainty"``: float, the CoCoA-CoT score
            - ``"u_A"``: float, answer-level confidence
            - ``"u_R"``: float, reasoning coherence uncertainty
            - ``"u_cons_A"``: float, answer semantic consistency
            - ``"greedy_answer"``: str
            - ``"greedy_chain"``: str
            - ``"sampled_answers"``: list[str]
            - ``"sampled_chains"``: list[str]
            - ``"alpha"``: float
        """
        M = M or self.M

        # ── Greedy decoding ───────────────────────────────────────────────────
        greedy_out = self.model.generate_greedy(prompt)

        # ── Stochastic sampling ───────────────────────────────────────────────
        sampled_outs = self.model.generate_sample(
            prompt,
            M=M,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )

        # Extract text components
        greedy_chain = greedy_out.chain_text
        greedy_answer = greedy_out.answer_text
        sampled_chains = [o.chain_text for o in sampled_outs]
        sampled_answers = [o.answer_text for o in sampled_outs]

        # ── Compute three uncertainty components ──────────────────────────────
        u_A = self._compute_u_A(greedy_out)
        u_R = self._compute_u_R(greedy_chain, sampled_chains)
        u_cons_A = self._compute_u_cons_A(greedy_answer, sampled_answers)

        # ── Combine: Û = [α·u_A + (1-α)·u_R] · Û_cons_A ─────────────────────
        combined_confidence = self.alpha * u_A + (1.0 - self.alpha) * u_R
        uncertainty = float(combined_confidence * u_cons_A)

        return {
            "uncertainty": uncertainty,
            "u_A": float(u_A),
            "u_R": float(u_R),
            "u_cons_A": float(u_cons_A),
            "greedy_answer": greedy_answer,
            "greedy_chain": greedy_chain,
            "sampled_answers": sampled_answers,
            "sampled_chains": sampled_chains,
            "alpha": self.alpha,
        }

    def estimate_batch(
        self, prompts: list[str], M: Optional[int] = None
    ) -> list[dict]:
        """Batch UQ estimation for multiple prompts.

        Args:
            prompts: List of input prompts.
            M: Number of samples per prompt.

        Returns:
            List of result dictionaries, one per prompt.
        """
        return [self.estimate(prompt, M=M) for prompt in prompts]

    # ── Component computation ─────────────────────────────────────────────────

    def _compute_u_A(self, gen_output: GenerationOutput) -> float:
        """Compute answer-level confidence u_A.

        Dispatches to MSP, PPL, or MTE estimator over answer tokens.

        Args:
            gen_output: Greedy :class:`~cocoa_cot.models.GenerationOutput`.

        Returns:
            u_A value (higher = more uncertain).
        """
        return float(self._conf_estimator.estimate(gen_output))

    def _compute_u_R(
        self, greedy_chain: str, sampled_chains: list[str]
    ) -> float:
        """Compute reasoning coherence uncertainty u_R.

        u_R(c* | x) = 1 - (1/M) Σᵢ s_step(c*, c(i))

        Uses the step-aligned similarity function.

        Args:
            greedy_chain: The greedy-decoded reasoning chain c*.
            sampled_chains: M sampled reasoning chains.

        Returns:
            u_R value in approximately ``[0, 1]``. Higher = more uncertain.
        """
        if not greedy_chain or not sampled_chains:
            return 0.0

        # Batch all step-aligned similarity calls
        similarities = self.step_sim.compute_batch(greedy_chain, sampled_chains)
        # Clamp to [eps, 1-eps] before computing u_R
        similarities_clamped = [
            max(_EPS, min(1.0 - _EPS, s)) for s in similarities
        ]
        mean_sim = float(np.mean(similarities_clamped))
        return 1.0 - mean_sim

    def _compute_u_cons_A(
        self, greedy_answer: str, sampled_answers: list[str]
    ) -> float:
        """Compute answer semantic consistency Û_cons_A.

        Û_cons_A(a* | x) = 1 - (1/M) Σᵢ s(a*, a(i))

        Args:
            greedy_answer: The greedy answer a*.
            sampled_answers: M sampled answers.

        Returns:
            Û_cons_A in approximately ``[0, 1]``. Higher = more uncertain.
        """
        if not sampled_answers:
            return 1.0

        reference = greedy_answer or ""
        candidates = [a for a in sampled_answers if a]

        if not candidates:
            return 1.0

        pairs = [(reference, c) for c in candidates]
        raw_scores = self.answer_sim.compute_batch(pairs)
        # Clamp to [eps, 1-eps]
        clamped = [max(_EPS, min(1.0 - _EPS, s)) for s in raw_scores]
        mean_sim = float(np.mean(clamped))
        return 1.0 - mean_sim

    # ── Black-box variant ─────────────────────────────────────────────────────

    def estimate_blackbox(
        self, greedy_answer: str, sampled_answers: list[str]
    ) -> float:
        """Black-box estimate: Û_BB = Û_cons_A.

        Uses only answer text strings (no logits/hidden states required).

        Args:
            greedy_answer: The greedy answer a*.
            sampled_answers: M sampled answers.

        Returns:
            Black-box uncertainty score.
        """
        return self._compute_u_cons_A(greedy_answer, sampled_answers)
