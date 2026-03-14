"""
Consistency-based uncertainty estimators: DegMat and Û_cons.

These estimators measure uncertainty by comparing semantic similarity
across M sampled answers (or full outputs in non-CoT mode).

Reference:
    Vashurin et al. (2025) arXiv:2502.04964
"""

from __future__ import annotations

import logging

import numpy as np

from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.uncertainty.base import BaseUQEstimator

logger = logging.getLogger(__name__)


class DegreeMatrixEstimator(BaseUQEstimator):
    """Degree Matrix uncertainty estimator.

    Computes uncertainty as:

        U_DegMat(x) = 1 - trace(D(x)) / M²

    where D is the diagonal of the pairwise similarity matrix.

    Since D[i,i] = s(y(i), y(i)) = 1 (self-similarity), this simplifies to
    measuring how far the average self-similarity is from 1/M.

    The original formulation computes the normalised graph degree:

        U_DegMat = 1 - (1/M²) Σᵢ Σⱼ s(y(i), y(j))

    Args:
        similarity_fn: :class:`~cocoa_cot.similarity.BaseSimilarity` instance.
    """

    def __init__(self, similarity_fn: BaseSimilarity) -> None:
        self.sim = similarity_fn

    def estimate(self, answers: list[str], similarity_fn: BaseSimilarity = None) -> float:
        """Estimate degree-matrix uncertainty from a set of answers.

        Args:
            answers: List of M sampled answer strings.
            similarity_fn: Optional override for the similarity function.

        Returns:
            Uncertainty score in approximately ``[0, 1]``.
        """
        sim = similarity_fn or self.sim
        M = len(answers)
        if M <= 1:
            return 0.0

        # Build all (i, j) pairs (including i==j for the diagonal)
        pairs = [
            (answers[i], answers[j])
            for i in range(M)
            for j in range(M)
        ]
        scores = sim.compute_batch(pairs)
        score_matrix = np.array(scores).reshape(M, M)

        # U_DegMat = 1 - mean of all pairwise similarities
        return float(1.0 - score_matrix.mean())


class ConsistencyEstimator(BaseUQEstimator):
    """MBR-grounded consistency uncertainty estimator.

    Computes:

        Û_cons(y* | x) = 1 - (1/M) Σᵢ s(y*, y(i))

    where y* is the evaluated sequence (e.g., greedy output) and y(i) are M
    stochastic samples.

    This is the CoCoA building block for answer-level consistency.

    Args:
        similarity_fn: :class:`~cocoa_cot.similarity.BaseSimilarity` instance.
    """

    def __init__(self, similarity_fn: BaseSimilarity) -> None:
        self.sim = similarity_fn

    def estimate(
        self,
        reference: str,
        candidates: list[str],
        similarity_fn: BaseSimilarity = None,
    ) -> float:
        """Estimate consistency uncertainty.

        Args:
            reference: The evaluated sequence y* (e.g., greedy answer).
            candidates: List of M sampled strings.
            similarity_fn: Optional override for the similarity function.

        Returns:
            Uncertainty score in approximately ``[0, 1]``. Higher = more
            uncertain.
        """
        sim = similarity_fn or self.sim
        if not candidates:
            return 1.0

        # Batch all (reference, candidate) pairs
        pairs = [(reference, c) for c in candidates]
        scores = sim.compute_batch(pairs)
        mean_sim = float(np.mean(scores))
        return 1.0 - mean_sim
