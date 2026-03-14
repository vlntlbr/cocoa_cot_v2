"""
StepSegmenter: splits a reasoning chain string into a list of step strings.

Strategy (in priority order):
1. Explicit markers: "Step 1:", "Step 2:", numbered/bulleted lines
2. Sentence-boundary splitting (regex-based, no spaCy dependency)
3. Newline splitting, filtering empty strings
4. Fallback: entire chain as single step
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Regex patterns for explicit step markers ───────────────────────────────────
_EXPLICIT_STEP_PATTERNS = [
    # "Step 1:", "Step 2:", etc.
    re.compile(r"(?:^|\n)\s*[Ss]tep\s+\d+[:\.\)]\s*", re.MULTILINE),
    # Numbered lists: "1.", "1)", "(1)", "1:"
    re.compile(r"(?:^|\n)\s*(?:\(\d+\)|\d+[\.\):])\s+", re.MULTILINE),
    # Bullet points: "•", "-", "*" at line start
    re.compile(r"(?:^|\n)\s*[•\-\*]\s+", re.MULTILINE),
    # "First,", "Second,", "Third,", ... "Finally,"
    re.compile(
        r"(?:^|\n)\s*(?:First|Second|Third|Fourth|Fifth|Sixth|"
        r"Seventh|Eighth|Ninth|Tenth|Finally|Lastly)[,:\s]",
        re.MULTILINE | re.IGNORECASE,
    ),
]

# Sentence-ending punctuation for fallback splitting
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"])")


class StepSegmenter:
    """
    Segments a reasoning chain into discrete reasoning steps.

    Steps are returned as non-empty strings.  At minimum one step is always
    returned (the full chain itself) — the segmenter never raises.

    Usage::

        seg = StepSegmenter()
        steps = seg.segment("Step 1: Compute x. Step 2: Multiply by 2.")
        # → ["Compute x.", "Multiply by 2."]
    """

    # ── Public API ────────────────────────────────────────────────────────────

    def segment(self, chain: str) -> list[str]:
        """Segment a reasoning chain string into a list of step strings.

        Args:
            chain: Full reasoning chain text.

        Returns:
            Non-empty list of step strings.  Each step is a stripped,
            non-empty string.
        """
        if not chain or not chain.strip():
            return [chain] if chain else [""]

        chain = chain.strip()

        # Strategy 1: explicit markers
        steps = self._try_explicit_markers(chain)
        if len(steps) > 1:
            return steps

        # Strategy 2: sentence boundaries
        steps = self._try_sentence_split(chain)
        if len(steps) > 1:
            return steps

        # Strategy 3: newline split
        steps = self._try_newline_split(chain)
        if len(steps) > 1:
            return steps

        # Fallback: single step
        return [chain]

    def segment_batch(self, chains: list[str]) -> list[list[str]]:
        """Segment a list of reasoning chains.

        Args:
            chains: List of chain strings.

        Returns:
            List of step-lists, one per input chain.
        """
        return [self.segment(chain) for chain in chains]

    # ── Strategy implementations ──────────────────────────────────────────────

    def _try_explicit_markers(self, chain: str) -> list[str]:
        """Split on explicit step markers (numbered lists, bullets, etc.)."""
        for pattern in _EXPLICIT_STEP_PATTERNS:
            splits = pattern.split(chain)
            # First element may be empty if chain starts with a marker
            steps = [s.strip() for s in splits if s and s.strip()]
            if len(steps) >= 2:
                return steps
        return [chain]

    def _try_sentence_split(self, chain: str) -> list[str]:
        """Split on sentence boundaries."""
        sentences = _SENTENCE_END.split(chain)
        steps = [s.strip() for s in sentences if s and s.strip()]
        return steps if steps else [chain]

    def _try_newline_split(self, chain: str) -> list[str]:
        """Split on double or single newlines."""
        # Try double-newline first (paragraph split)
        parts = re.split(r"\n{2,}", chain)
        steps = [p.strip() for p in parts if p and p.strip()]
        if len(steps) >= 2:
            return steps

        # Single newline split
        parts = chain.splitlines()
        steps = [p.strip() for p in parts if p and p.strip()]
        return steps if steps else [chain]
