"""
Abstract base class for all LLM wrappers in CoCoA-CoT.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np


@dataclass
class GenerationOutput:
    """Structured output from a single LLM generation.

    Attributes:
        text: Full generated text (including chain and answer).
        token_ids: List of generated token IDs.
        token_logprobs: Log p(token | context) for each generated token
            (natural log, in the same order as ``token_ids``).
        token_entropies: Per-position entropy H(p_t) of the vocabulary
            distribution.  Same length as ``token_ids``.
        chain_text: Extracted reasoning chain (everything before the answer).
        answer_text: Extracted final answer string.
        answer_token_logprobs: Log-probs restricted to answer tokens only.
        answer_token_entropies: Entropies restricted to answer tokens only.
    """

    text: str
    token_ids: list[int] = field(default_factory=list)
    token_logprobs: list[float] = field(default_factory=list)
    token_entropies: list[float] = field(default_factory=list)
    chain_text: str = ""
    answer_text: str = ""
    answer_token_logprobs: list[float] = field(default_factory=list)
    answer_token_entropies: list[float] = field(default_factory=list)


class BaseModel(ABC):
    """Abstract interface for LLM wrappers.

    Both white-box (HuggingFace) and black-box (API) models implement this
    interface so that uncertainty estimators are model-agnostic.
    """

    @abstractmethod
    def generate_greedy(self, prompt: str) -> GenerationOutput:
        """Generate a single greedy-decoded output.

        Args:
            prompt: The input prompt string.

        Returns:
            A :class:`GenerationOutput` with token log-probs and entropies.
        """
        ...

    @abstractmethod
    def generate_sample(
        self,
        prompt: str,
        M: int,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 1.0,
    ) -> list[GenerationOutput]:
        """Generate M stochastic samples.

        Args:
            prompt: The input prompt string.
            M: Number of samples to generate.
            temperature: Sampling temperature.
            top_k: Top-k truncation parameter.
            top_p: Nucleus sampling parameter.

        Returns:
            List of M :class:`GenerationOutput` objects.
        """
        ...

    def get_hidden_states(
        self, prompt: str, layer_idx: int
    ) -> "Optional[np.ndarray]":
        """Extract hidden states at a specific layer.

        Default implementation returns ``None`` (not supported for black-box
        models).  Override in white-box subclasses.

        Args:
            prompt: The input prompt.
            layer_idx: Layer index (0-indexed).

        Returns:
            Array of shape ``(seq_len, d_model)`` or ``None``.
        """
        return None
