"""
Black-box model wrapper for CoCoA-CoT.

Simulates the black-box evaluation scenario by:
1. Accepting pre-generated text outputs (no access to logits/hidden states)
2. Providing only text-level generation outputs
3. Setting token_logprobs and token_entropies to empty lists

Also supports API-based generation via a simple text completion interface
(e.g., OpenAI-compatible endpoints).
"""

from __future__ import annotations

import logging
from typing import Callable, Optional

from cocoa_cot.models.base import BaseModel, GenerationOutput
from cocoa_cot.parsing.chain_parser import ChainParser

logger = logging.getLogger(__name__)


class BlackBoxModel(BaseModel):
    """Black-box LLM wrapper.

    In black-box mode, token-level statistics (log-probs, entropies) are
    unavailable.  Only the text output is accessible.

    This is used in :mod:`cocoa_cot.experiments.run_blackbox` to evaluate
    the text-only consistency estimator ``Û_BB = Û_cons_A``.

    Args:
        generation_fn: A callable ``f(prompt: str) -> str`` for single
            generation.  Can be an API wrapper.
        parser: :class:`~cocoa_cot.parsing.ChainParser` instance.
        max_new_tokens: Passed to generation_fn if it accepts it.
    """

    def __init__(
        self,
        generation_fn: Optional[Callable[[str], str]] = None,
        parser: Optional[ChainParser] = None,
        max_new_tokens: int = 512,
    ) -> None:
        self.generation_fn = generation_fn
        self.parser = parser or ChainParser()
        self.max_new_tokens = max_new_tokens

    # ── Factory: wrap a white-box model stripping internal stats ─────────────

    @classmethod
    def from_hf_model(cls, hf_model: "HFModel") -> "BlackBoxModel":
        """Create a black-box wrapper around an HFModel.

        The returned model generates text using the underlying HF model but
        discards all logits and hidden states, keeping only the text.

        Args:
            hf_model: White-box :class:`~cocoa_cot.models.HFModel` instance.

        Returns:
            A :class:`BlackBoxModel` wrapping the HF model.
        """

        def gen_fn(prompt: str) -> str:
            out = hf_model.generate_greedy(prompt)
            return out.text

        bb = cls(generation_fn=gen_fn, parser=hf_model.parser)
        return bb

    # ── BaseModel interface ───────────────────────────────────────────────────

    def generate_greedy(self, prompt: str) -> GenerationOutput:
        """Generate a single output without token statistics.

        Args:
            prompt: Input prompt string.

        Returns:
            :class:`GenerationOutput` with empty token_logprobs and
            token_entropies.
        """
        if self.generation_fn is None:
            raise RuntimeError("No generation_fn provided to BlackBoxModel.")

        text = self.generation_fn(prompt)
        chain_text, answer_text = self.parser.parse(text, format="auto")
        return GenerationOutput(
            text=text,
            token_ids=[],
            token_logprobs=[],
            token_entropies=[],
            chain_text=chain_text,
            answer_text=answer_text,
            answer_token_logprobs=[],
            answer_token_entropies=[],
        )

    def generate_sample(
        self,
        prompt: str,
        M: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> list[GenerationOutput]:
        """Generate M samples without token statistics.

        Args:
            prompt: Input prompt string.
            M: Number of samples.
            temperature: Sampling temperature (passed to generation_fn if
                supported; otherwise ignored).
            top_k: Top-k parameter.
            top_p: Nucleus parameter.

        Returns:
            List of M :class:`GenerationOutput` objects (text only).
        """
        if self.generation_fn is None:
            raise RuntimeError("No generation_fn provided to BlackBoxModel.")

        outputs = []
        for _ in range(M):
            text = self.generation_fn(prompt)
            chain_text, answer_text = self.parser.parse(text, format="auto")
            outputs.append(
                GenerationOutput(
                    text=text,
                    token_ids=[],
                    token_logprobs=[],
                    token_entropies=[],
                    chain_text=chain_text,
                    answer_text=answer_text,
                    answer_token_logprobs=[],
                    answer_token_entropies=[],
                )
            )
        return outputs

    # get_hidden_states returns None (inherited from BaseModel default)
