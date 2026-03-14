"""
Hybrid uncertainty estimators: SemanticEntropy and SAR.

SemanticEntropy:
    Groups sampled answers into semantic clusters using bidirectional NLI
    entailment.  Computes entropy over cluster probability distribution.

SAR (Sentence-level Answer Relevance):
    SAR = u_A(y*) · Û_cons(y*) where u_A uses TokenSAR weighting.
    Token relevance: R_T(y_k) = 1 - s(x ∪ y, x ∪ y\y_k)

References:
    Semantic Entropy: Kuhn et al. (2023)
    SAR: Duan et al. (2023)
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.special import logsumexp

from cocoa_cot.models.base import GenerationOutput
from cocoa_cot.similarity.base import BaseSimilarity
from cocoa_cot.similarity.nli import NLISimilarity
from cocoa_cot.uncertainty.base import BaseUQEstimator

logger = logging.getLogger(__name__)


class SemanticEntropyEstimator(BaseUQEstimator):
    """Semantic Entropy uncertainty estimator.

    Clusters sampled answers using bidirectional NLI entailment, then
    computes Shannon entropy over cluster probabilities weighted by the
    log-probabilities of cluster members.

    Cluster probability:
        p(C_k | x) ∝ Σ_{y ∈ C_k} p(y | x) = Σ_{y ∈ C_k} exp(log_p(y))

    Semantic entropy:
        SE(x) = -Σ_k p(C_k | x) log p(C_k | x)

    Args:
        nli_fn: :class:`~cocoa_cot.similarity.NLISimilarity` instance for
            bidirectional entailment clustering.
    """

    def __init__(self, nli_fn: NLISimilarity) -> None:
        self.nli_fn = nli_fn

    def estimate(
        self,
        answers: list[str],
        log_probs: list[float],
        nli_fn: NLISimilarity = None,
    ) -> float:
        """Estimate semantic entropy.

        Args:
            answers: List of M sampled answer strings.
            log_probs: Log p(y(i) | x) for each sampled answer.  These are
                total sequence log-probs (sum of token log-probs).
            nli_fn: Optional NLI function override.

        Returns:
            Semantic entropy score (higher = more uncertain).
        """
        nli = nli_fn or self.nli_fn
        M = len(answers)
        if M <= 1:
            return 0.0

        # Cluster answers using bidirectional entailment
        labels = nli.cluster_by_entailment(answers)
        n_clusters = max(labels) + 1

        # Aggregate log-probs per cluster (log-sum-exp for numerical stability)
        cluster_log_probs: dict[int, list[float]] = {k: [] for k in range(n_clusters)}
        for i, label in enumerate(labels):
            lp = log_probs[i] if i < len(log_probs) else 0.0
            cluster_log_probs[label].append(lp)

        # Normalised cluster probabilities
        raw_cluster_log_p = np.array(
            [logsumexp(v) for v in cluster_log_probs.values()], dtype=np.float64
        )
        log_Z = logsumexp(raw_cluster_log_p)
        normalised_log_p = raw_cluster_log_p - log_Z  # log p(C_k | x)
        cluster_probs = np.exp(normalised_log_p)

        # Shannon entropy
        entropy = -float(np.sum(cluster_probs * normalised_log_p))
        return entropy


class SAREstimator(BaseUQEstimator):
    """SAR (Sentence-level Answer Relevance) uncertainty estimator.

    SAR = TokenSAR(y*) · Û_cons(y*)

    TokenSAR weights each token's contribution to uncertainty by how much
    removing that token changes the semantic similarity to the full output:

        R_T(y_k) = 1 - s(x + y, x + y\\{y_k})
        TokenSAR = Σ_k R_T(y_k) · (-log p(y_k | context))

    For efficiency, we approximate token relevance using cosine similarity
    of sentence embeddings rather than a full cross-encoder per token.

    Args:
        similarity_fn: :class:`~cocoa_cot.similarity.BaseSimilarity` instance.
    """

    def __init__(self, similarity_fn: BaseSimilarity) -> None:
        self.sim = similarity_fn

    def estimate(
        self,
        gen_output: GenerationOutput,
        candidates: list[str],
        similarity_fn: BaseSimilarity = None,
    ) -> float:
        """Estimate SAR uncertainty.

        Args:
            gen_output: Greedy :class:`~cocoa_cot.models.GenerationOutput`.
            candidates: M sampled answer strings (for consistency component).
            similarity_fn: Optional similarity override.

        Returns:
            SAR uncertainty score (higher = more uncertain).
        """
        sim = similarity_fn or self.sim

        # ── TokenSAR component ────────────────────────────────────────────────
        token_sar = self._compute_token_sar(gen_output, sim)

        # ── Consistency component Û_cons ─────────────────────────────────────
        reference = gen_output.answer_text or gen_output.text
        if candidates:
            pairs = [(reference, c) for c in candidates]
            raw_scores = sim.compute_batch(pairs)
            u_cons = 1.0 - float(np.mean(raw_scores))
        else:
            u_cons = 0.0

        return float(token_sar * u_cons)

    def _compute_token_sar(
        self,
        gen_output: GenerationOutput,
        sim: BaseSimilarity,
    ) -> float:
        """Compute the TokenSAR component.

        Uses an approximation: relevance of token k is estimated as
        ``1 - mean_sim(full_answer, answer_minus_k)``.

        For computational efficiency, we approximate using PPL when the
        answer is very short, else compute leave-one-token-out similarity.
        """
        answer = gen_output.answer_text or gen_output.text
        logprobs = gen_output.answer_token_logprobs or gen_output.token_logprobs

        if not logprobs:
            return 0.0

        if len(logprobs) <= 3:
            # Too short to compute meaningful relevance; fall back to NLL
            return float(-np.mean(logprobs))

        words = answer.split()
        if len(words) <= 1:
            return float(-np.mean(logprobs))

        # Build leave-one-out pairs: (full, answer_minus_word_k)
        loo_texts = [
            " ".join(words[:k] + words[k + 1 :]) for k in range(len(words))
        ]
        pairs = [(answer, loo) for loo in loo_texts]
        sims = sim.compute_batch(pairs)
        relevances = np.array([1.0 - s for s in sims], dtype=np.float64)

        # Align relevances to token log-probs (words ↔ tokens; use uniform weight)
        n_tokens = len(logprobs)
        n_words = len(words)
        # Simple repeat/truncate to match lengths
        if n_words != n_tokens:
            indices = np.round(np.linspace(0, n_words - 1, n_tokens)).astype(int)
            relevances = relevances[indices]

        neg_logprobs = -np.array(logprobs, dtype=np.float64)
        # Normalise relevances to sum to 1 for weighting
        rel_sum = relevances.sum()
        if rel_sum > 1e-9:
            weights = relevances / rel_sum
        else:
            weights = np.ones(n_tokens) / n_tokens

        return float(np.sum(weights * neg_logprobs))
