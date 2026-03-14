"""
Task-specific quality measures for all 6 benchmark datasets.

Each function takes a model prediction string and a gold reference string
and returns a quality score in [0, 1].
"""

from __future__ import annotations

import logging
import re
import subprocess
import sys
import tempfile
import textwrap
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Numeric extraction helper ─────────────────────────────────────────────────

_NUM_RE = re.compile(r"-?\d+(?:[.,]\d+)*")


def _extract_number(text: str) -> Optional[float]:
    """Extract the last number from a text string."""
    text = text.replace(",", "")
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


# ── GSM8K / exact-match accuracy ─────────────────────────────────────────────

def gsm8k_accuracy(prediction: str, gold: str) -> float:
    """Exact numeric accuracy for GSM8K (and ARC / ProntoQA exact match).

    Extracts the final number from both strings and compares them.
    Falls back to case-insensitive string comparison.

    Args:
        prediction: Model's predicted answer string.
        gold: Gold reference answer string.

    Returns:
        1.0 if answers match, 0.0 otherwise.
    """
    # Try numeric comparison
    pred_num = _extract_number(prediction)
    gold_num = _extract_number(gold)

    if pred_num is not None and gold_num is not None:
        return 1.0 if abs(pred_num - gold_num) < 1e-6 else 0.0

    # Fallback: normalised string comparison
    pred_norm = _normalize_str(prediction)
    gold_norm = _normalize_str(gold)
    return 1.0 if pred_norm == gold_norm else 0.0


def _normalize_str(s: str) -> str:
    """Normalise a string: lowercase, strip whitespace, remove punctuation."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# ── MATH-500 accuracy ─────────────────────────────────────────────────────────

def math500_accuracy(prediction: str, gold: str) -> float:
    """Symbolic accuracy for MATH-500 competition problems.

    Tries three approaches in order:
    1. SymPy symbolic equality (parse_latex)
    2. Numeric floating-point comparison
    3. Normalised string equality

    Args:
        prediction: Model's predicted answer string.
        gold: Gold reference answer string (may be LaTeX).

    Returns:
        1.0 if answers match, 0.0 otherwise.
    """
    # 1. SymPy symbolic comparison
    try:
        from sympy import simplify, latex, N
        from sympy.parsing.latex import parse_latex
        from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application

        def _try_parse(s: str):
            s = s.strip().replace("\\\\", "\\")
            # Try LaTeX first
            try:
                return parse_latex(s)
            except Exception:
                pass
            # Try sympy expr
            try:
                transforms = standard_transformations + (implicit_multiplication_application,)
                return parse_expr(s, transformations=transforms)
            except Exception:
                return None

        pred_expr = _try_parse(prediction)
        gold_expr = _try_parse(gold)

        if pred_expr is not None and gold_expr is not None:
            try:
                diff = simplify(pred_expr - gold_expr)
                if diff == 0:
                    return 1.0
                # Numerical check
                pred_val = complex(N(pred_expr))
                gold_val = complex(N(gold_expr))
                if abs(pred_val - gold_val) < 1e-6:
                    return 1.0
            except Exception:
                pass
    except ImportError:
        logger.warning("sympy not available; using string comparison for math500")
    except Exception as exc:
        logger.debug("math500_accuracy sympy failed: %s", exc)

    # 2. Numeric comparison
    pred_num = _extract_number(prediction)
    gold_num = _extract_number(gold)
    if pred_num is not None and gold_num is not None:
        return 1.0 if abs(pred_num - gold_num) < 1e-6 else 0.0

    # 3. Normalised string comparison
    return 1.0 if _normalize_str(prediction) == _normalize_str(gold) else 0.0


# ── AlignScore / cosine-proxy ─────────────────────────────────────────────────

def alignscore(
    prediction: str,
    reference: str,
    model=None,
) -> float:
    """Alignment score using sentence-transformer cosine similarity as proxy.

    If a sentence-transformer model is provided, uses it directly.
    Otherwise instantiates a lightweight default model.

    Args:
        prediction: Model's predicted answer string.
        reference: Gold reference answer string.
        model: Optional pre-loaded SentenceTransformer model.

    Returns:
        Cosine similarity in [0, 1].
    """
    if not prediction or not reference:
        return 0.0

    try:
        from sentence_transformers import SentenceTransformer, util

        _model = model
        if _model is None:
            # Cache a module-level default model to avoid repeated loading
            if not hasattr(alignscore, "_default_model"):
                alignscore._default_model = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2"
                )
            _model = alignscore._default_model

        pred_emb = _model.encode(prediction, convert_to_tensor=True)
        ref_emb = _model.encode(reference, convert_to_tensor=True)
        cos_sim = float(util.cos_sim(pred_emb, ref_emb).item())
        # Clip to [0, 1]
        return max(0.0, min(1.0, (cos_sim + 1.0) / 2.0))
    except ImportError:
        logger.warning("sentence_transformers not available; falling back to ROUGE-L")
        from cocoa_cot.similarity.lexical import RougeL
        return RougeL().compute(prediction, reference)
    except Exception as exc:
        logger.warning("alignscore failed: %s", exc)
        return 0.0


# ── LiveCodeBench pass@1 ──────────────────────────────────────────────────────

_PYTHON_CODE_RE = re.compile(
    r"```(?:python)?\s*\n?(.*?)```", re.DOTALL | re.IGNORECASE
)


def _extract_code(text: str) -> str:
    """Extract Python code from markdown code blocks, or return as-is."""
    matches = _PYTHON_CODE_RE.findall(text)
    if matches:
        return matches[-1].strip()
    return text.strip()


def livecodebench_pass_at_1(
    prediction: str,
    test_cases: list[dict],
    timeout: float = 5.0,
) -> float:
    """Execute prediction against test cases in a sandboxed subprocess.

    IMPORTANT: Never uses exec() or eval() directly. All code execution
    happens in an isolated subprocess with a timeout.

    Args:
        prediction: Model-generated Python code string.
        test_cases: List of dicts with ``"input"`` and ``"output"`` keys.
        timeout: Maximum execution time per test case (seconds).

    Returns:
        1.0 if all test cases pass, 0.0 otherwise.
    """
    code = _extract_code(prediction)
    if not code:
        return 0.0

    for tc in test_cases:
        tc_input = str(tc.get("input", ""))
        expected_output = str(tc.get("output", "")).strip()

        # Build a complete Python script that reads stdin and writes stdout
        test_script = textwrap.dedent(f"""
import sys

# ----- user code -----
{code}
# ----- end user code -----
""")
        try:
            result = subprocess.run(
                [sys.executable, "-c", test_script],
                input=tc_input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            actual_output = result.stdout.strip()
            if actual_output != expected_output:
                return 0.0
        except subprocess.TimeoutExpired:
            return 0.0
        except Exception as exc:
            logger.debug("livecodebench_pass_at_1 error: %s", exc)
            return 0.0

    return 1.0


# ── Dispatch ──────────────────────────────────────────────────────────────────

def get_quality_fn(dataset_name: str) -> Callable[[str, Any], float]:
    """Return the appropriate quality function for a dataset.

    Args:
        dataset_name: One of ``"gsm8k"``, ``"math500"``, ``"hotpotqa"``,
            ``"arc_challenge"``, ``"prontoqa"``, ``"livecodebench"``.

    Returns:
        Quality function ``f(prediction, gold) -> float`` in [0, 1].

    Raises:
        KeyError: If dataset_name is not recognised.
    """
    dispatch: dict[str, Callable] = {
        "gsm8k": gsm8k_accuracy,
        "math500": math500_accuracy,
        "hotpotqa": alignscore,
        "arc_challenge": gsm8k_accuracy,
        "prontoqa": gsm8k_accuracy,
        "livecodebench": livecodebench_pass_at_1,
    }
    if dataset_name not in dispatch:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {list(dispatch.keys())}"
        )
    return dispatch[dataset_name]
