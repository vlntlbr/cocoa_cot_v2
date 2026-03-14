"""
Original CoCoA estimator (Vashurin et al., 2025).

Implements the baseline CoCoA formula applied to full outputs (not split
into chain + answer):

    Û_CoCoA(y* | x) = u(y* | x) · (1/M) Σᵢ (1 − s(y*, y(i)))

where u is one of MSP, PPL, or MTE.

This is used as a strong baseline comparison in Table 1.
"""

from __future__ import annotations

import logging

import numpy as np

from cocoa_cot.models.base import GenerationOutput
from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.uncertainty.base import BaseUQEstimator
from cocoa_cot.uncertainty.information import MSPEstimator, PPLEstimator, MTEEstimator

logger = logging.getLogger(__name__)

_CONFIDENCE_MAP = {
    "msp": MSPEstimator,
    "ppl": PPLEstimator,
    "mte": MTEEstimator,
}


class CoCoAEstimator(BaseUQEstimator):
    """Original CoCoA uncertainty estimator.

    Û_CoCoA = u(y*|x) · (1/M) Σᵢ (1 − s(y*, y(i)))

    Note: applied to full output y* (not decomposed into chain + answer).
    This means confidence u is computed over all tokens.

    Args:
        confidence_type: One of ``"msp"``, ``"ppl"``, or ``"mte"``.
        similarity_fn: :class:`~cocoa_cot.similarity.BaseSimilarity` instance.
    """

    def __init__(
        self,
        confidence_type: str,
        similarity_fn: BaseSimilarity,
    ) -> None:
        if confidence_type not in _CONFIDENCE_MAP:
            raise ValueError(
                f"confidence_type must be one of {list(_CONFIDENCE_MAP)}, "
                f"got {confidence_type!r}"
            )
        self.confidence_type = confidence_type
        self.sim = similarity_fn
        # Use non-CoT mode (all tokens)
        conf_cls = _CONFIDENCE_MAP[confidence_type]
        self._conf_estimator = conf_cls(cot_mode=False)

    def estimate(
        self,
        gen_output: GenerationOutput,
        sampled_outputs: list[GenerationOutput],
    ) -> float:
        """Estimate original CoCoA uncertainty.

        Args:
            gen_output: Greedy-decoded :class:`~cocoa_cot.models.GenerationOutput`.
            sampled_outputs: List of M sampled
                :class:`~cocoa_cot.models.GenerationOutput` objects.

        Returns:
            CoCoA uncertainty score (higher = more uncertain).
        """
        # Confidence term u(y*|x)
        u = self._conf_estimator.estimate(gen_output)

        # Consistency term (1/M) Σᵢ (1 - s(y*, y(i)))
        reference = gen_output.text
        candidates = [o.text for o in sampled_outputs]

        if not candidates:
            return float(u)

        pairs = [(reference, c) for c in candidates]
        scores = self.sim.compute_batch(pairs)
        u_cons = float(np.mean([1.0 - s for s in scores]))

        return float(u * u_cons)
