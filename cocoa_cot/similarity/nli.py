"""
NLI-based similarity using a DeBERTa cross-encoder trained for NLI.

The similarity score is defined as the probability of the "entailment" class:
    s_NLI(a, b) = P(entailment | a, b)

For bidirectional NLI (SemanticEntropy clustering), both
directions are checked.

Default model: ``cross-encoder/nli-deberta-v3-large``
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from cocoa_cot.similarity.base import BaseSimilarity

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-large"

# Class order returned by the DeBERTa NLI cross-encoder
# (contradiction, neutral, entailment)
_ENTAIL_IDX = 2
_CONTRA_IDX = 0


class NLISimilarity(BaseSimilarity):
    """NLI-based semantic similarity.

    Scores are the softmax probability of the *entailment* class.

    Args:
        model_name: HuggingFace model identifier for NLI cross-encoder.
        batch_size: Batch size for inference.
        device: Compute device.
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
        self._model = None

    def _get_model(self):
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading NLI CrossEncoder: %s", self.model_name)
                self._model = CrossEncoder(
                    self.model_name,
                    device=self._device,
                    max_length=512,
                )
            except Exception as exc:
                logger.error(
                    "Failed to load NLI CrossEncoder %s: %s", self.model_name, exc
                )
                raise
        return self._model

    # ── BaseSimilarity interface ─────────────────────────────────────────────

    def compute(self, text_a: str, text_b: str) -> float:
        """Compute NLI-based similarity for a single pair."""
        scores = self.compute_batch([(text_a, text_b)])
        return scores[0]

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute NLI entailment probability for a batch of pairs.

        Args:
            pairs: List of ``(premise, hypothesis)`` tuples.

        Returns:
            List of entailment probabilities in ``[0, 1]``.
        """
        if not pairs:
            return []

        try:
            model = self._get_model()
            # Returns (N, 3) logits: [contradiction, neutral, entailment]
            logits = model.predict(
                pairs,
                batch_size=self.batch_size,
                show_progress_bar=False,
                apply_softmax=True,
            )
            logits = np.asarray(logits, dtype=np.float64)
            if logits.ndim == 1:
                logits = logits.reshape(1, -1)
            # Return entailment probability
            return logits[:, _ENTAIL_IDX].tolist()
        except Exception as exc:
            logger.warning(
                "NLI inference failed (%s). Falling back to ROUGE-L.", exc
            )
            from cocoa_cot.similarity.lexical import RougeL

            fallback = RougeL()
            return fallback.compute_batch(pairs)

    # ── Clustering helpers ───────────────────────────────────────────────────

    def bidirectional_entail(self, text_a: str, text_b: str) -> bool:
        """Return True if both a→b and b→a are entailment.

        Used in SemanticEntropy clustering:
        ``p_entail(a, b) > p_contra(a, b)``  AND
        ``p_entail(b, a) > p_contra(b, a)``

        Args:
            text_a: First sentence.
            text_b: Second sentence.

        Returns:
            True if mutually entailed, False otherwise.
        """
        model = self._get_model()
        pairs = [(text_a, text_b), (text_b, text_a)]
        logits = model.predict(
            pairs,
            batch_size=2,
            show_progress_bar=False,
            apply_softmax=True,
        )
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)
        ab_entail = logits[0, _ENTAIL_IDX] > logits[0, _CONTRA_IDX]
        ba_entail = logits[1, _ENTAIL_IDX] > logits[1, _CONTRA_IDX]
        return bool(ab_entail and ba_entail)

    def cluster_by_entailment(
        self, texts: list[str]
    ) -> list[int]:
        """Cluster texts using bidirectional NLI entailment.

        Each pair of texts that mutually entail each other is placed in the
        same cluster.  Uses a greedy union-find approach.

        Args:
            texts: List of answer strings.

        Returns:
            List of integer cluster labels, one per text.
        """
        n = len(texts)
        # Build all bidirectional pairs at once for efficiency
        forward_pairs = [
            (texts[i], texts[j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        backward_pairs = [(b, a) for a, b in forward_pairs]

        if not forward_pairs:
            return list(range(n))

        model = self._get_model()
        all_pairs = forward_pairs + backward_pairs
        logits = model.predict(
            all_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            apply_softmax=True,
        )
        logits = np.asarray(logits, dtype=np.float64)
        if logits.ndim == 1:
            logits = logits.reshape(1, -1)

        n_pairs = len(forward_pairs)
        fwd_logits = logits[:n_pairs]
        bwd_logits = logits[n_pairs:]

        # Union-Find
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            parent[find(x)] = find(y)

        for idx, (i, j) in enumerate(
            (i, j) for i in range(n) for j in range(i + 1, n)
        ):
            fwd_e = fwd_logits[idx, _ENTAIL_IDX] > fwd_logits[idx, _CONTRA_IDX]
            bwd_e = bwd_logits[idx, _ENTAIL_IDX] > bwd_logits[idx, _CONTRA_IDX]
            if fwd_e and bwd_e:
                union(i, j)

        # Remap to contiguous labels
        root_to_label: dict[int, int] = {}
        labels = []
        for i in range(n):
            root = find(i)
            if root not in root_to_label:
                root_to_label[root] = len(root_to_label)
            labels.append(root_to_label[root])
        return labels
