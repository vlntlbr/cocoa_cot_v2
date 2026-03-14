"""Tests for ChainParser — all formats, edge cases, answer extraction."""

import pytest
from cocoa_cot.parsing.chain_parser import ChainParser


@pytest.fixture()
def parser() -> ChainParser:
    return ChainParser()


# ── DeepSeek format ───────────────────────────────────────────────────────────

class TestDeepSeekFormat:
    def test_basic(self, parser):
        raw = "<think>Step 1. Compute 2+2.\nStep 2. That equals 4.</think><answer>4</answer>"
        chain, answer = parser.parse(raw, format="deepseek")
        assert "Step 1" in chain
        assert answer == "4"

    def test_multiline_chain(self, parser):
        raw = "<think>\nLine A\nLine B\nLine C\n</think><answer>   42   </answer>"
        chain, answer = parser.parse(raw, format="deepseek")
        assert "Line A" in chain
        assert answer == "42"

    def test_whitespace_stripped(self, parser):
        raw = "<think>   reasoning   </think><answer>  yes  </answer>"
        _, answer = parser.parse(raw, format="deepseek")
        assert answer == "yes"

    def test_missing_think_tag_falls_through(self, parser):
        """Without <think>, auto-format should still return a result."""
        raw = "<answer>42</answer>"
        chain, answer = parser.parse(raw, format="auto")
        assert answer == "42"

    def test_nested_content_preserved(self, parser):
        raw = "<think>f(x) = x^2 + 1\nTherefore f(3) = 10.</think><answer>10</answer>"
        chain, answer = parser.parse(raw, format="deepseek")
        assert "f(x)" in chain
        assert answer == "10"


# ── GSM8K format ──────────────────────────────────────────────────────────────

class TestGSM8KFormat:
    def test_basic(self, parser):
        raw = "We calculate 5 * 6 = 30.\n#### 30"
        chain, answer = parser.parse(raw, format="gsm8k")
        assert "calculate" in chain
        assert answer == "30"

    def test_answer_with_spaces(self, parser):
        raw = "Reasoning here.\n####   $123   "
        _, answer = parser.parse(raw, format="gsm8k")
        assert "123" in answer

    def test_no_separator_returns_whole_text(self, parser):
        raw = "Just some reasoning without a separator."
        chain, answer = parser.parse(raw, format="gsm8k")
        # Should not crash; some answer must be returned
        assert isinstance(answer, str)
        assert isinstance(chain, str)

    def test_multiple_hashes_takes_last(self, parser):
        raw = "Step 1: #### intermediate\nStep 2: more work\n#### 99"
        _, answer = parser.parse(raw, format="gsm8k")
        assert "99" in answer


# ── Llama CoT format ──────────────────────────────────────────────────────────

class TestLlamaCoTFormat:
    def test_therefore_marker(self, parser):
        raw = "Let me think. x = 5. Therefore, the answer is 5."
        chain, answer = parser.parse(raw, format="llama_cot")
        assert isinstance(chain, str)
        assert "5" in answer

    def test_so_the_answer_marker(self, parser):
        raw = "We add 3 and 4 to get 7. So the answer is 7."
        _, answer = parser.parse(raw, format="llama_cot")
        assert "7" in answer

    def test_case_insensitive(self, parser):
        raw = "Computing... THE ANSWER IS: 42"
        _, answer = parser.parse(raw, format="llama_cot")
        assert "42" in answer


# ── Auto format ───────────────────────────────────────────────────────────────

class TestAutoFormat:
    def test_auto_picks_deepseek(self, parser):
        raw = "<think>Chain.</think><answer>7</answer>"
        chain, answer = parser.parse(raw, format="auto")
        assert answer == "7"

    def test_auto_picks_gsm8k(self, parser):
        raw = "Work shown here.\n#### 99"
        _, answer = parser.parse(raw, format="auto")
        assert "99" in answer

    def test_auto_fallback_non_empty(self, parser):
        raw = "Just plain text output with no known separator."
        chain, answer = parser.parse(raw, format="auto")
        assert isinstance(chain, str)
        assert isinstance(answer, str)
        assert len(chain) + len(answer) > 0

    def test_empty_input_does_not_crash(self, parser):
        chain, answer = parser.parse("", format="auto")
        assert isinstance(chain, str)
        assert isinstance(answer, str)

    def test_only_whitespace(self, parser):
        chain, answer = parser.parse("   \n\t  ", format="auto")
        assert isinstance(chain, str)
        assert isinstance(answer, str)


# ── Answer char-offset helper ─────────────────────────────────────────────────

class TestAnswerCharOffsets:
    def test_offsets_within_raw(self, parser):
        raw = "<think>Some chain text.</think><answer>42</answer>"
        start, end = parser.get_answer_char_offsets(raw, format="deepseek")
        assert start >= 0
        assert end > start
        assert raw[start:end].strip() == "42"

    def test_offsets_none_when_no_answer(self, parser):
        raw = "No answer marker here."
        result = parser.get_answer_char_offsets(raw, format="deepseek")
        # Either None or a valid (start, end) tuple
        if result is not None:
            start, end = result
            assert end >= start
