"""
Dataset loaders for all 6 CoCoA-CoT benchmarks.

Supported datasets:
    - gsm8k        (GSM8K math word problems)
    - math500      (MATH-500 competition math)
    - hotpotqa     (HotPotQA multi-hop QA)
    - arc_challenge (ARC-Challenge science MCQ)
    - prontoqa     (ProntoQA logical reasoning)
    - livecodebench (LiveCodeBench code generation)

All loaders return a dict with:
    "eval":    list of {"prompt": str, "answer": str, "extra": dict}
    "holdout": list of {"prompt": str, "answer": str, "extra": dict}
"""

from __future__ import annotations

import logging
import random
from typing import Any

logger = logging.getLogger(__name__)

# ── Prompt templates ──────────────────────────────────────────────────────────

_PROMPTS = {
    "gsm8k": (
        "Solve the following math problem step by step.\n"
        "Problem: {question}\nSolution:"
    ),
    "math500": (
        "Solve the following math problem. Show your reasoning.\n"
        "Problem: {problem}\nSolution:"
    ),
    "hotpotqa": (
        "Answer the following question by reasoning step by step.\n"
        "Question: {question}\nContext: {context}\nAnswer:"
    ),
    "arc_challenge": (
        "Answer the following science question by reasoning step by step.\n"
        "Question: {question}\nChoices: {choices}\nAnswer:"
    ),
    "prontoqa": (
        "Using the given facts, determine if the statement is true or false. "
        "Show your reasoning.\nFacts: {facts}\nStatement: {statement}\nAnswer:"
    ),
    "livecodebench": (
        "Solve the following programming problem.\n"
        "{problem_statement}\n\nWrite a Python solution:"
    ),
}


def load_dataset_splits(
    dataset_name: str,
    n_eval: int = 500,
    n_holdout: int = 2000,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Load a dataset and return eval and holdout splits.

    Args:
        dataset_name: One of the 6 supported dataset names.
        n_eval: Number of examples for main evaluation.
        n_holdout: Number of examples for CoCoA-CoT Light training.
        seed: Random seed for reproducible sampling.

    Returns:
        Dictionary with keys ``"eval"`` and ``"holdout"``, each containing a
        list of records ``{"prompt": str, "answer": str, "extra": dict}``.

    Raises:
        ValueError: If dataset_name is not recognised.
    """
    loaders = {
        "gsm8k": _load_gsm8k,
        "math500": _load_math500,
        "hotpotqa": _load_hotpotqa,
        "arc_challenge": _load_arc_challenge,
        "prontoqa": _load_prontoqa,
        "livecodebench": _load_livecodebench,
    }
    if dataset_name not in loaders:
        raise ValueError(
            f"Unknown dataset {dataset_name!r}. "
            f"Available: {list(loaders.keys())}"
        )

    logger.info("Loading dataset: %s (n_eval=%d, n_holdout=%d, seed=%d)",
                dataset_name, n_eval, n_holdout, seed)
    return loaders[dataset_name](n_eval=n_eval, n_holdout=n_holdout, seed=seed)


# ── GSM8K ─────────────────────────────────────────────────────────────────────

def _load_gsm8k(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("gsm8k", "main", split="test")
    ds_train = load_dataset("gsm8k", "main", split="train")

    rng = random.Random(seed)
    eval_indices = _sample_indices(len(ds), n_eval, rng)
    holdout_indices = _sample_indices(len(ds_train), n_holdout, rng)

    template = _PROMPTS["gsm8k"]

    def _make_record(example: dict) -> dict:
        prompt = template.format(question=example["question"])
        # Gold answer is after "####" in the answer field
        answer = example["answer"].split("####")[-1].strip()
        return {"prompt": prompt, "answer": answer, "extra": {"question": example["question"]}}

    eval_split = [_make_record(ds[i]) for i in eval_indices]
    holdout_split = [_make_record(ds_train[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


# ── MATH-500 ──────────────────────────────────────────────────────────────────

def _load_math500(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    # Try the standard MATH dataset from lighteval
    try:
        ds = load_dataset("lighteval/MATH", "all", split="test")
    except Exception:
        ds = load_dataset("hendrycks/competition_math", split="test")

    rng = random.Random(seed)
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)
    eval_indices = all_indices[:n_eval]
    holdout_indices = all_indices[n_eval : n_eval + n_holdout]

    template = _PROMPTS["math500"]

    def _make_record(example: dict) -> dict:
        problem = example.get("problem") or example.get("question", "")
        solution = example.get("solution") or example.get("answer", "")
        # Extract boxed answer if present
        answer = _extract_boxed(solution) or solution
        prompt = template.format(problem=problem)
        return {"prompt": prompt, "answer": answer, "extra": {"level": example.get("level", "")}}

    eval_split = [_make_record(ds[i]) for i in eval_indices]
    holdout_split = [_make_record(ds[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


def _extract_boxed(text: str) -> str:
    """Extract the content of \\boxed{} from a math solution."""
    import re
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    return m.group(1).strip() if m else ""


# ── HotPotQA ──────────────────────────────────────────────────────────────────

def _load_hotpotqa(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    ds = load_dataset("hotpot_qa", "distractor", split="validation")

    rng = random.Random(seed)
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)
    eval_indices = all_indices[:n_eval]
    holdout_indices = all_indices[n_eval : n_eval + n_holdout]

    template = _PROMPTS["hotpotqa"]

    def _make_record(example: dict) -> dict:
        question = example["question"]
        answer = example["answer"]
        # Flatten context (list of paragraphs)
        context_items = example.get("context", {})
        if isinstance(context_items, dict):
            titles = context_items.get("title", [])
            sentences = context_items.get("sentences", [])
            context = " ".join(
                f"{t}: {''.join(s)}" for t, s in zip(titles, sentences)
            )[:1000]  # truncate to 1000 chars
        else:
            context = str(context_items)[:1000]

        prompt = template.format(question=question, context=context)
        return {"prompt": prompt, "answer": answer, "extra": {"type": example.get("type", "")}}

    eval_split = [_make_record(ds[i]) for i in eval_indices]
    holdout_split = [_make_record(ds[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


# ── ARC-Challenge ─────────────────────────────────────────────────────────────

def _load_arc_challenge(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    ds_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    ds_train = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="train")

    rng = random.Random(seed)
    eval_indices = _sample_indices(len(ds_test), n_eval, rng)
    holdout_indices = _sample_indices(len(ds_train), n_holdout, rng)

    template = _PROMPTS["arc_challenge"]

    def _make_record(example: dict) -> dict:
        question = example["question"]
        # Format choices as "A. ..., B. ..., ..."
        choices = example.get("choices", {})
        if isinstance(choices, dict):
            labels = choices.get("label", [])
            texts = choices.get("text", [])
            choices_str = ", ".join(f"{l}. {t}" for l, t in zip(labels, texts))
        else:
            choices_str = str(choices)
        answer = example.get("answerKey", "")
        prompt = template.format(question=question, choices=choices_str)
        return {"prompt": prompt, "answer": answer, "extra": {"choices": choices_str}}

    eval_split = [_make_record(ds_test[i]) for i in eval_indices]
    holdout_split = [_make_record(ds_train[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


# ── ProntoQA ──────────────────────────────────────────────────────────────────

def _load_prontoqa(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset("EleutherAI/prontoqa", split="test")
        ds_train = load_dataset("EleutherAI/prontoqa", split="train")
    except Exception:
        # Fallback: use a split of the test set
        ds_full = load_dataset("EleutherAI/prontoqa", split="train")
        rng_split = random.Random(seed)
        all_idx = list(range(len(ds_full)))
        rng_split.shuffle(all_idx)
        mid = len(all_idx) // 2
        ds = ds_full.select(all_idx[:mid])
        ds_train = ds_full.select(all_idx[mid:])

    rng = random.Random(seed)
    eval_indices = _sample_indices(len(ds), n_eval, rng)
    holdout_indices = _sample_indices(len(ds_train), n_holdout, rng)

    template = _PROMPTS["prontoqa"]

    def _make_record(example: dict) -> dict:
        # ProntoQA has varying field names; try multiple
        facts = (
            example.get("facts")
            or example.get("context")
            or example.get("story", "")
        )
        statement = (
            example.get("statement")
            or example.get("query")
            or example.get("question", "")
        )
        answer = (
            example.get("answer")
            or example.get("label")
            or ("True" if example.get("correct", True) else "False")
        )
        if isinstance(facts, list):
            facts = " ".join(str(f) for f in facts)
        prompt = template.format(facts=str(facts), statement=str(statement))
        return {"prompt": prompt, "answer": str(answer), "extra": {}}

    eval_split = [_make_record(ds[i]) for i in eval_indices]
    holdout_split = [_make_record(ds_train[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


# ── LiveCodeBench ─────────────────────────────────────────────────────────────

def _load_livecodebench(
    n_eval: int = 500, n_holdout: int = 2000, seed: int = 42
) -> dict:
    from datasets import load_dataset

    try:
        ds = load_dataset("livecodebench/code_generation_lite", split="test")
    except Exception:
        ds = load_dataset("livecodebench/code_generation_lite", split="train")

    rng = random.Random(seed)
    all_indices = list(range(len(ds)))
    rng.shuffle(all_indices)
    eval_indices = all_indices[:n_eval]
    holdout_indices = all_indices[n_eval : n_eval + n_holdout]

    template = _PROMPTS["livecodebench"]

    def _make_record(example: dict) -> dict:
        problem_stmt = (
            example.get("question_content")
            or example.get("problem_statement")
            or example.get("question", "")
        )
        solutions = example.get("solutions") or example.get("solution") or ""
        test_cases = example.get("public_test_cases") or example.get("test_cases") or []
        if isinstance(test_cases, str):
            import json
            try:
                test_cases = json.loads(test_cases)
            except Exception:
                test_cases = []
        prompt = template.format(problem_statement=problem_stmt)
        return {
            "prompt": prompt,
            "answer": str(solutions),
            "extra": {"test_cases": test_cases},
        }

    eval_split = [_make_record(ds[i]) for i in eval_indices]
    holdout_split = [_make_record(ds[i]) for i in holdout_indices]

    return {"eval": eval_split, "holdout": holdout_split}


# ── Utility ───────────────────────────────────────────────────────────────────

def _sample_indices(n_total: int, n_sample: int, rng: random.Random) -> list[int]:
    """Sample n_sample indices from 0..n_total-1 without replacement."""
    n_sample = min(n_sample, n_total)
    indices = list(range(n_total))
    rng.shuffle(indices)
    return indices[:n_sample]
