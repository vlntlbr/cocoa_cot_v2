"""
Abstract base class for uncertainty quantification estimators.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class BaseUQEstimator(ABC):
    """Abstract interface for all UQ estimators.

    All estimators follow the same pattern:
    - Accept a :class:`~cocoa_cot.models.GenerationOutput` (or similar) as input
    - Return a scalar uncertainty score (higher = more uncertain)
    """

    @abstractmethod
    def estimate(self, *args, **kwargs) -> float:
        """Estimate uncertainty score.

        Returns:
            Scalar uncertainty score. Higher values indicate more uncertainty.
        """
        ...
