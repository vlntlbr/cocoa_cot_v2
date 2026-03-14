"""
Cross-encoder semantic similarity using sentence-transformers.

Default model: ``cross-encoder/stsb-roberta-large``
Scores are normalised to [0, 1] via sigmoid.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from cocoa_cot.similarity.base import BaseSimilarity

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/stsb-roberta-large"


class CrossEncoderSimilarity(BaseSimilarity):
    """Semantic similarity using a cross-encoder model.

    Scores are produced by the cross-encoder and then passed through
    ``sigmoid`` to map them into ``[0, 1]``.

    Args:
        model_name: HuggingFace model identifier.
        batch_size: Batch size for ``CrossEncoder.predict()``.
        device: ``"cuda"``, ``"cpu"``, or ``None`` (auto-detect).
    """

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL,
        batch_size: int = 32,
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._device = device
        self._model = None  # Lazy load

    # ── Lazy model loading ───────────────────────────────────────────────────

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading CrossEncoder: %s", self.model_name)
                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                    max_length=512,
                )
            except Exception as exc:
                logger.error("Failed to load CrossEncoder %s: %s", self.model_name, exc)
                raise
        return self._model

    # ── BaseSimilarity interface ─────────────────────────────────────────────

    def compute(self, text_a: str, text_b: str) -> float:
        """Compute similarity for a single pair.

        Args:
            text_a: First text.
            text_b: Second text.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        scores = self.compute_batch([(text_a, text_b)])
        return scores[0]

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute similarity for a batch of pairs.

        All pairs are processed in a single batched cross-encoder call for
        efficiency.

        Args:
            pairs: List of ``(text_a, text_b)`` tuples.

        Returns:
            List of similarity scores in ``[0, 1]``.
        """
        if not pairs:
            return []

        try:
            model = self._get_model()
            raw_scores = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
            )
            # Map to [0, 1] via sigmoid
            scores = _sigmoid(np.asarray(raw_scores, dtype=np.float64))
            return scores.tolist()
        except Exception as exc:
            logger.warning(
                "CrossEncoder inference failed (%s). Falling back to ROUGE-L.", exc
            )
            from cocoa_cot.similarity.lexical import RougeL

            fallback = RougeL()
            return fallback.compute_batch(pairs)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))
