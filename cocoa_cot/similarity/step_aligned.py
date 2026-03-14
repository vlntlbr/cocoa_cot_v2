"""
Step-aligned similarity: s_step(c, c') for comparing reasoning chains.

Implements Eq. 11 from the paper:
    s_step(c, c') = (1/K) Σₖ max_{k'} s_sem(cₖ, c'_{k'})

This is the core novel similarity function of CoCoA-CoT.
It is robust to chains of different lengths and step permutations.

Note: s_step is ASYMMETRIC by design (normalised by K, the length of chain c).
"""

from __future__ import annotations

import logging

import numpy as np

from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.parsing.step_segmenter import StepSegmenter

logger = logging.getLogger(__name__)


class StepAlignedSimilarity:
    """Step-aligned similarity between reasoning chains.

    Computes ``s_step(c, c') = (1/K) Σₖ max_{k'} s_sem(cₖ, c'_{k'})``
    where cₖ are steps of the reference chain and c'_{k'} are steps of the
    candidate chain.

    Args:
        base_similarity: Underlying semantic similarity function (e.g.,
            :class:`~cocoa_cot.similarity.CrossEncoderSimilarity`).
        segmenter: :class:`~cocoa_cot.parsing.StepSegmenter` instance used to
            split chains into steps.
    """

    def __init__(
        self,
        base_similarity: BaseSimilarity,
        segmenter: StepSegmenter,
    ) -> None:
        self.sim = base_similarity
        self.seg = segmenter

    # ── Public API ────────────────────────────────────────────────────────────

    def compute(self, chain_a: str, chain_b: str) -> float:
        """Compute s_step(chain_a, chain_b).

        Args:
            chain_a: Reference chain (normalisation is over K steps of this).
            chain_b: Candidate chain.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        steps_a = self.seg.segment(chain_a)
        steps_b = self.seg.segment(chain_b)
        return self._compute_from_steps(steps_a, steps_b)

    def compute_batch(
        self,
        reference_chain: str,
        candidate_chains: list[str],
    ) -> list[float]:
        """Compute s_step(reference_chain, c) for all c in candidate_chains.

        All cross-encoder calls are batched together for efficiency.

        Args:
            reference_chain: The greedy-decoded chain (reference).
            candidate_chains: M sampled chains.

        Returns:
            List of similarity scores in ``[0, 1]``, one per candidate.
        """
        if not candidate_chains:
            return []

        # Segment once up front
        ref_steps = self.seg.segment(reference_chain)
        cand_steps_list = [self.seg.segment(c) for c in candidate_chains]

        K = len(ref_steps)

        # Build a flat list of all (step_k, step_k') pairs across all candidates
        # and record metadata for reconstruction
        all_pairs: list[tuple[str, str]] = []
        pair_indices: list[tuple[int, int, int]] = []  # (cand_idx, k, k')

        for cand_idx, cand_steps in enumerate(cand_steps_list):
            for k, step_a in enumerate(ref_steps):
                for kp, step_b in enumerate(cand_steps):
                    all_pairs.append((step_a, step_b))
                    pair_indices.append((cand_idx, k, kp))

        if not all_pairs:
            return [0.0] * len(candidate_chains)

        # Single batched similarity call
        all_scores = self.sim.compute_batch(all_pairs)

        # Reconstruct per-candidate score matrices and compute max-alignment
        # Shape: [n_candidates, K, K'_max] — but K' varies, so use dicts
        from collections import defaultdict

        # score_matrix[cand_idx][k][k'] = score
        score_matrix: dict[int, dict[int, dict[int, float]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        for (cand_idx, k, kp), score in zip(pair_indices, all_scores):
            score_matrix[cand_idx][k][kp] = score

        results = []
        for cand_idx, cand_steps in enumerate(cand_steps_list):
            step_max_scores = []
            for k in range(K):
                kp_scores = score_matrix[cand_idx].get(k, {})
                if kp_scores:
                    step_max_scores.append(max(kp_scores.values()))
                else:
                    step_max_scores.append(0.0)
            results.append(float(np.mean(step_max_scores)))

        return results

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _compute_from_steps(
        self, steps_a: list[str], steps_b: list[str]
    ) -> float:
        """Compute s_step from pre-segmented step lists.

        Args:
            steps_a: Reference steps (K of them).
            steps_b: Candidate steps.

        Returns:
            Similarity score in ``[0, 1]``.
        """
        K = len(steps_a)
        Kp = len(steps_b)

        # Build full cross-product pair list for a single batched call
        pairs = [
            (steps_a[k], steps_b[kp])
            for k in range(K)
            for kp in range(Kp)
        ]

        scores_flat = self.sim.compute_batch(pairs)

        # Reshape to (K, Kp) and take row-wise max
        score_matrix = np.array(scores_flat, dtype=np.float64).reshape(K, Kp)
        best_per_step = score_matrix.max(axis=1)  # shape (K,)
        return float(best_per_step.mean())
