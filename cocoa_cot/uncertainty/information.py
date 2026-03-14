"""
Information-based uncertainty estimators: MSP, PPL, MTE.

These are token-probability baselines that measure uncertainty from the
model's own confidence in its generated output.

Reference equations:
    u_MSP(y*|x) = -Σ_t log p(y*_t | y*_{<t}, x)       [total NLL]
    u_PPL(y*|x) = -(1/L) Σ_t log p(y*_t | y*_{<t}, x)  [per-token NLL]
    u_MTE(y*|x) = (1/L) Σ_t H(p_t)                      [mean token entropy]

When applied in CoT mode, these use *answer tokens only* (answer_token_logprobs /
answer_token_entropies).  In standard mode they use all tokens.
"""

from __future__ import annotations

import logging

import numpy as np

from cocoa_cot.models.base import GenerationOutput
from cocoa_cot.uncertainty.base import BaseUQEstimator

logger = logging.getLogger(__name__)

_EPS = 1e-9  # numerical guard


class MSPEstimator(BaseUQEstimator):
    """Maximum Sequence Probability estimator.

    Computes total negative log-likelihood of the generated sequence:

        u_MSP = -Σ_t log p(t | context)

    In CoT mode uses answer tokens only.

    Args:
        cot_mode: If True, compute over answer tokens only.
    """

    def __init__(self, cot_mode: bool = True) -> None:
        self.cot_mode = cot_mode

    def estimate(self, gen_output: GenerationOutput) -> float:
        """Estimate MSP uncertainty.

        Args:
            gen_output: A :class:`~cocoa_cot.models.GenerationOutput` with
                populated ``token_logprobs`` and ``answer_token_logprobs``.

        Returns:
            Total NLL (higher = more uncertain).
        """
        logprobs = self._get_logprobs(gen_output)
        if not logprobs:
            return 0.0
        return float(-np.sum(logprobs))

    def _get_logprobs(self, gen_output: GenerationOutput) -> list[float]:
        if self.cot_mode and gen_output.answer_token_logprobs:
            return gen_output.answer_token_logprobs
        return gen_output.token_logprobs


class PPLEstimator(BaseUQEstimator):
    """Perplexity-based uncertainty estimator.

    Computes per-token negative log-likelihood (length-normalised NLL):

        u_PPL = -(1/L) Σ_t log p(t | context)

    In CoT mode uses answer tokens only.

    Args:
        cot_mode: If True, compute over answer tokens only.
    """

    def __init__(self, cot_mode: bool = True) -> None:
        self.cot_mode = cot_mode

    def estimate(self, gen_output: GenerationOutput) -> float:
        """Estimate PPL uncertainty.

        Args:
            gen_output: A :class:`~cocoa_cot.models.GenerationOutput`.

        Returns:
            Mean NLL per token (higher = more uncertain).
        """
        logprobs = self._get_logprobs(gen_output)
        if not logprobs:
            return 0.0
        return float(-np.mean(logprobs))

    def _get_logprobs(self, gen_output: GenerationOutput) -> list[float]:
        if self.cot_mode and gen_output.answer_token_logprobs:
            return gen_output.answer_token_logprobs
        return gen_output.token_logprobs


class MTEEstimator(BaseUQEstimator):
    """Mean Token Entropy estimator.

    Computes the average entropy of the vocabulary distribution at each
    generated token position:

        u_MTE = (1/L) Σ_t H(p_t)

    where H(p_t) = -Σ_v p_v log p_v.

    In CoT mode uses answer tokens only.

    Args:
        cot_mode: If True, compute over answer tokens only.
    """

    def __init__(self, cot_mode: bool = True) -> None:
        self.cot_mode = cot_mode

    def estimate(self, gen_output: GenerationOutput) -> float:
        """Estimate MTE uncertainty.

        Args:
            gen_output: A :class:`~cocoa_cot.models.GenerationOutput` with
                populated ``token_entropies`` and ``answer_token_entropies``.

        Returns:
            Mean token entropy (higher = more uncertain).
        """
        entropies = self._get_entropies(gen_output)
        if not entropies:
            return 0.0
        return float(np.mean(entropies))

    def _get_entropies(self, gen_output: GenerationOutput) -> list[float]:
        if self.cot_mode and gen_output.answer_token_entropies:
            return gen_output.answer_token_entropies
        return gen_output.token_entropies
