"""
Abstract base class for all similarity functions used in CoCoA-CoT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseSimilarity(ABC):
    """Abstract interface for semantic similarity functions.

    All similarity implementations must inherit from this class and implement
    :meth:`compute` and :meth:`compute_batch`.

    Scores are always in the range ``[0, 1]`` where ``1.0`` means identical
    and ``0.0`` means maximally dissimilar.
    """

    @abstractmethod
    def compute(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two text strings.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        ...

    @abstractmethod
    def compute_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[float]:
        """Compute similarity for a batch of text pairs.

        Args:
            pairs: List of ``(text_a, text_b)`` tuples.

        Returns:
            List of similarity scores in ``[0, 1]``, one per pair.
        """
        ...

    def compute_one_to_many(
        self, reference: str, candidates: list[str]
    ) -> list[float]:
        """Compute similarity from one reference to many candidates.

        Default implementation batches all pairs and calls :meth:`compute_batch`.

        Args:
            reference: The reference text (e.g., greedy answer).
            candidates: List of candidate texts (e.g., sampled answers).

        Returns:
            List of similarity scores, one per candidate.
        """
        pairs = [(reference, c) for c in candidates]
        return self.compute_batch(pairs)
