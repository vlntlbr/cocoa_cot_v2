"""
ChainParser: extracts (reasoning_chain, answer) pairs from raw LLM outputs.

Supports multiple chain formats:
- DeepSeek-R1: <think>...</think><answer>...</answer>
- Few-shot CoT: "The answer is:", "Therefore,", "####"
- Generic: heuristic final-sentence split
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from cocoa_cot.parsing.step_segmenter import StepSegmenter

logger = logging.getLogger(__name__)

# ── Format-specific regexes ────────────────────────────────────────────────────
_DEEPSEEK_THINK = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    re.DOTALL | re.IGNORECASE,
)
_DEEPSEEK_THINK_OPEN = re.compile(r"<think>(.*)</think>(.*)", re.DOTALL | re.IGNORECASE)
_DEEPSEEK_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

_GSM8K_HASH = re.compile(r"(.*?)####\s*(.+)$", re.DOTALL)

_COT_MARKERS = [
    re.compile(r"(.*?)(?:therefore[,.]?\s+the answer is[:\s]+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:the answer is[:\s]+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:so the answer is[:\s]+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:thus,?\s+the answer is[:\s]+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:in conclusion[,.]?\s+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:\*\*answer\*\*[:\s]+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:answer:\s+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
    re.compile(r"(.*?)(?:final answer:\s+)(.*?)\.?\s*$", re.DOTALL | re.IGNORECASE),
]


class ChainParser:
    """
    Parses raw LLM generation into structured (chain, answer) pairs.

    Supports multiple chain formats:

    - ``"deepseek"``: uses ``<think>...</think><answer>...</answer>`` tags
    - ``"gsm8k"``: answer follows ``####``
    - ``"llama_cot"``: answer follows "The answer is:" / "Therefore,"
    - ``"auto"``: tries each format in order, returns first success

    If parsing fails entirely, returns (full_text, "") and logs a warning.
    """

    def __init__(self) -> None:
        self.segmenter = StepSegmenter()

    # ── Public API ──────────────────────────────────────────────────────────

    def parse(self, raw_output: str, format: str = "auto") -> tuple[str, str]:
        """Extract (reasoning_chain, answer) from a raw LLM generation.

        Args:
            raw_output: The full text produced by the language model.
            format: One of ``"deepseek"``, ``"gsm8k"``, ``"llama_cot"``,
                    or ``"auto"``.

        Returns:
            A tuple ``(chain, answer)`` where *chain* is the full reasoning
            text (everything before the answer) and *answer* is the minimal
            final answer string.  If parsing fails, chain equals raw_output
            and answer equals "".
        """
        raw_output = raw_output.strip()

        if format == "auto":
            for fmt in ("deepseek", "gsm8k", "llama_cot"):
                chain, answer = self._try_format(raw_output, fmt)
                if answer:
                    return chain, answer
            # Last-resort: split on last sentence
            return self._generic_split(raw_output)

        chain, answer = self._try_format(raw_output, format)
        if not answer:
            logger.warning(
                "ChainParser: format=%r failed for text (first 80 chars): %r. "
                "Returning full text as chain.",
                format,
                raw_output[:80],
            )
            return raw_output, ""
        return chain, answer

    def parse_batch(
        self, outputs: list[str], format: str = "auto"
    ) -> list[tuple[str, str]]:
        """Parse a batch of raw outputs.

        Args:
            outputs: List of raw LLM generations.
            format: Parsing format to use for all items.

        Returns:
            List of ``(chain, answer)`` tuples, one per output.
        """
        return [self.parse(text, format=format) for text in outputs]

    # ── Format-specific parsers ─────────────────────────────────────────────

    def _try_format(self, text: str, fmt: str) -> tuple[str, str]:
        """Attempt to parse ``text`` with ``fmt``; return ("", "") on failure."""
        if fmt == "deepseek":
            return self._parse_deepseek(text)
        if fmt == "gsm8k":
            return self._parse_gsm8k(text)
        if fmt == "llama_cot":
            return self._parse_llama_cot(text)
        return "", ""

    def _parse_deepseek(self, text: str) -> tuple[str, str]:
        """Parse DeepSeek-R1 style ``<think>…</think><answer>…</answer>``."""
        m = _DEEPSEEK_THINK.search(text)
        if m:
            chain = m.group(1).strip()
            answer = m.group(2).strip()
            return chain, answer

        # Unclosed <answer> tag
        m = _DEEPSEEK_ANSWER_TAG.search(text)
        if m:
            answer = m.group(1).strip()
            chain_end = text.find("<answer>")
            chain = text[:chain_end].strip()
            # Strip <think> wrapper if present
            chain = re.sub(r"^\s*<think>\s*", "", chain, flags=re.IGNORECASE)
            chain = re.sub(r"\s*</think>\s*$", "", chain, flags=re.IGNORECASE)
            return chain, answer

        return "", ""

    def _parse_gsm8k(self, text: str) -> tuple[str, str]:
        """Parse GSM8K style ``…#### <number>``."""
        m = _GSM8K_HASH.search(text)
        if m:
            chain = m.group(1).strip()
            answer = m.group(2).strip()
            return chain, answer
        return "", ""

    def _parse_llama_cot(self, text: str) -> tuple[str, str]:
        """Parse few-shot CoT style with natural-language answer markers."""
        for pattern in _COT_MARKERS:
            m = pattern.search(text)
            if m:
                chain = m.group(1).strip()
                answer = m.group(2).strip()
                if answer:
                    return chain, answer
        return "", ""

    def _generic_split(self, text: str) -> tuple[str, str]:
        """Heuristic fallback: last non-empty line becomes the answer."""
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if len(lines) >= 2:
            return "\n".join(lines[:-1]), lines[-1]
        if lines:
            # Single line — use last sentence
            sentences = re.split(r"(?<=[.!?])\s+", lines[0])
            if len(sentences) >= 2:
                return " ".join(sentences[:-1]), sentences[-1]
            return lines[0], lines[0]
        return text, ""

    # ── Offset mapping helpers ──────────────────────────────────────────────

    def get_answer_char_offsets(
        self, raw_output: str, format: str = "auto"
    ) -> Optional[tuple[int, int]]:
        """Return the (start, end) character offsets of the answer span.

        Used by HFModel to map answer characters → token positions via the
        tokenizer's offset_mapping.

        Returns None if the answer could not be located.
        """
        chain, answer = self.parse(raw_output, format=format)
        if not answer:
            return None
        # Find the *last* occurrence to handle duplicates in the chain
        idx = raw_output.rfind(answer)
        if idx == -1:
            return None
        return idx, idx + len(answer)
