"""
Lexical similarity functions: ROUGE-L and BLEU.

Used as fallbacks when neural similarity models are unavailable,
and as baselines in ablation experiments.
"""

from __future__ import annotations

import logging

import numpy as np

from cocoa_cot.similarity.base import BaseSimilarity

logger = logging.getLogger(__name__)


class RougeL(BaseSimilarity):
    """ROUGE-L F-score similarity.

    Scores are in ``[0, 1]`` where 1.0 means identical sequences.

    Uses the ``rouge_score`` library.
    """

    def __init__(self) -> None:
        self._scorer = None

    def _get_scorer(self):
        if self._scorer is None:
            from rouge_score import rouge_scorer

            self._scorer = rouge_scorer.RougeScorer(
                ["rougeL"], use_stemmer=False
            )
        return self._scorer

    def compute(self, text_a: str, text_b: str) -> float:
        """Compute ROUGE-L F1 between two texts."""
        scorer = self._get_scorer()
        result = scorer.score(text_a, text_b)
        return float(result["rougeL"].fmeasure)

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute ROUGE-L F1 for a batch of pairs."""
        scorer = self._get_scorer()
        return [
            float(scorer.score(a, b)["rougeL"].fmeasure)
            for a, b in pairs
        ]


class BLEU(BaseSimilarity):
    """Sentence-level BLEU similarity.

    Uses smoothed BLEU-4 (SacreBLEU style) normalised to ``[0, 1]``.
    """

    def compute(self, text_a: str, text_b: str) -> float:
        """Compute BLEU between reference ``text_a`` and hypothesis ``text_b``."""
        return self.compute_batch([(text_a, text_b)])[0]

    def compute_batch(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Compute BLEU for a batch of pairs."""
        try:
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

            smoother = SmoothingFunction().method1
            scores = []
            for ref, hyp in pairs:
                ref_tokens = ref.lower().split()
                hyp_tokens = hyp.lower().split()
                if not hyp_tokens:
                    scores.append(0.0)
                    continue
                score = sentence_bleu(
                    [ref_tokens],
                    hyp_tokens,
                    smoothing_function=smoother,
                )
                scores.append(float(score))
            return scores
        except ImportError:
            logger.warning("nltk not available; falling back to ROUGE-L for BLEU.")
            fallback = RougeL()
            return fallback.compute_batch(pairs)
