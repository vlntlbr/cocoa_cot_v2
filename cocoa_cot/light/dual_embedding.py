"""
Dual-embedding feature extractor for CoCoA-CoT Light.

Extracts mean-pooled hidden states from:
    e_c: reasoning chain tokens (at a specified transformer layer)
    e_a: answer tokens (at the same layer)

These are concatenated as input to the AuxiliaryModel MLP.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class DualEmbeddingExtractor:
    """Extract dual (chain, answer) embeddings from a white-box HF model.

    For each input, extracts hidden states at layer ``layer_idx`` and
    mean-pools over:
    - Chain tokens → e_c of shape (d_model,)
    - Answer tokens → e_a of shape (d_model,)

    The concatenated [e_c; e_a] ∈ R^{2*d_model} is the feature vector for
    the AuxiliaryModel.

    Args:
        model: :class:`~cocoa_cot.models.HFModel` white-box model.
        layer_idx: Transformer layer index for hidden state extraction.
    """

    def __init__(self, model: "HFModel", layer_idx: int = 16) -> None:  # type: ignore[name-defined]
        self.model = model
        self.layer_idx = layer_idx

    def extract(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        """Extract (e_c, e_a) embeddings for a single prompt.

        The model generates a greedy output, then hidden states are extracted
        for the full sequence.  Token positions are split into chain vs answer
        based on the parser's output.

        Args:
            prompt: Input prompt string.

        Returns:
            Tuple of ``(e_c, e_a)`` where each is a 1D numpy array of shape
            ``(d_model,)``.
        """
        # Greedy generate to get the text and chain/answer split
        gen_out = self.model.generate_greedy(prompt)
        chain_text = gen_out.chain_text
        answer_text = gen_out.answer_text
        full_text = gen_out.text

        # Extract hidden states for the entire generated context (prompt + output)
        full_context = prompt + full_text
        hs = self.model.get_hidden_states(full_context, self.layer_idx)
        # hs: (seq_len, d_model)

        if hs is None or hs.ndim < 2:
            d = getattr(self.model._model, "config", None)
            d = getattr(d, "hidden_size", 4096) if d else 4096
            return np.zeros(d), np.zeros(d)

        seq_len, d_model = hs.shape

        # Identify chain and answer token spans by text search
        e_c = self._mean_pool_span(hs, full_context, chain_text, seq_len)
        e_a = self._mean_pool_span(hs, full_context, answer_text, seq_len)

        return e_c, e_a

    def extract_batch(
        self, prompts: list[str]
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Extract dual embeddings for a batch of prompts.

        Args:
            prompts: List of input prompt strings.

        Returns:
            List of ``(e_c, e_a)`` tuples.
        """
        return [self.extract(p) for p in prompts]

    def _mean_pool_span(
        self,
        hidden_states: np.ndarray,
        full_text: str,
        span_text: str,
        seq_len: int,
    ) -> np.ndarray:
        """Mean-pool hidden states over token positions corresponding to span_text.

        Uses a heuristic character-fraction approach to identify token range.
        """
        d_model = hidden_states.shape[1]

        if not span_text:
            return hidden_states.mean(axis=0)

        # Find character position of span in full_text
        start_char = full_text.find(span_text)
        if start_char == -1:
            # Fall back to mean-pooling all tokens
            return hidden_states.mean(axis=0)

        end_char = start_char + len(span_text)
        total_chars = max(len(full_text), 1)

        # Approximate token positions by character fraction
        start_tok = int(start_char / total_chars * seq_len)
        end_tok = int(end_char / total_chars * seq_len)

        start_tok = max(0, min(start_tok, seq_len - 1))
        end_tok = max(start_tok + 1, min(end_tok, seq_len))

        span_hs = hidden_states[start_tok:end_tok]
        if span_hs.shape[0] == 0:
            return hidden_states.mean(axis=0)

        return span_hs.mean(axis=0).astype(np.float32)
