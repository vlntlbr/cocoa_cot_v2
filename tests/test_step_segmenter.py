"""Tests for StepSegmenter — numbered steps, markers, sentence fallback, edge cases."""

import pytest
from cocoa_cot.parsing.step_segmenter import StepSegmenter


@pytest.fixture()
def seg() -> StepSegmenter:
    return StepSegmenter()


# ── Explicit numbered steps ───────────────────────────────────────────────────

class TestNumberedSteps:
    def test_step_n_colon(self, seg):
        chain = "Step 1: Compute 2+2.\nStep 2: The result is 4."
        steps = seg.segment(chain)
        assert len(steps) == 2
        assert "2+2" in steps[0]
        assert "result" in steps[1]

    def test_digit_dot(self, seg):
        chain = "1. First do A.\n2. Then do B.\n3. Finally do C."
        steps = seg.segment(chain)
        assert len(steps) == 3

    def test_digit_paren(self, seg):
        chain = "1) Alpha\n2) Beta\n3) Gamma"
        steps = seg.segment(chain)
        assert len(steps) == 3


# ── First/Second/Finally markers ──────────────────────────────────────────────

class TestOrdinalMarkers:
    def test_first_second_finally(self, seg):
        chain = "First, we add. Second, we subtract. Finally, we conclude."
        steps = seg.segment(chain)
        assert len(steps) >= 2  # at minimum two distinct parts

    def test_case_insensitive(self, seg):
        chain = "FIRST: step one. SECOND: step two."
        steps = seg.segment(chain)
        assert len(steps) >= 2


# ── Sentence boundary fallback ────────────────────────────────────────────────

class TestSentenceBoundaryFallback:
    def test_period_split(self, seg):
        chain = "We start here. We continue there. We end here."
        steps = seg.segment(chain)
        assert len(steps) >= 2

    def test_question_exclamation(self, seg):
        chain = "What is 2+2? It is 4! Then we are done."
        steps = seg.segment(chain)
        assert len(steps) >= 2


# ── Newline fallback ──────────────────────────────────────────────────────────

class TestNewlineFallback:
    def test_newline_split(self, seg):
        chain = "line one\nline two\nline three"
        steps = seg.segment(chain)
        assert len(steps) == 3

    def test_blank_lines_filtered(self, seg):
        chain = "step a\n\n\nstep b\n\nstep c"
        steps = seg.segment(chain)
        assert len(steps) == 3
        for s in steps:
            assert s.strip() != ""


# ── Edge cases ────────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_chain_returns_one_step(self, seg):
        steps = seg.segment("")
        assert len(steps) == 1

    def test_whitespace_only_returns_one_step(self, seg):
        steps = seg.segment("   \n\t  ")
        assert len(steps) == 1

    def test_single_sentence(self, seg):
        steps = seg.segment("The answer is 42.")
        assert len(steps) == 1
        assert "42" in steps[0]

    def test_never_returns_empty_list(self, seg):
        for chain in ["", " ", "\n", "abc"]:
            steps = seg.segment(chain)
            assert len(steps) >= 1

    def test_no_empty_step_strings(self, seg):
        chain = "Step 1: A\nStep 2: B\nStep 3: C"
        for step in seg.segment(chain):
            assert step.strip() != ""
