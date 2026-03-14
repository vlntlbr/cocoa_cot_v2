"""
Shared utilities for experiment scripts.

Provides:
- Config loading (YAML with base inheritance)
- Model / similarity / estimator factory
- Generation caching
- Result saving helpers
- Seed setting
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import random
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import yaml

logger = logging.getLogger(__name__)


# ── Config loading ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load a YAML config file, merging with its _base_ if present.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Merged configuration dictionary.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    base_key = cfg.pop("_base_", None)
    if base_key:
        base_path = Path(config_path).parent / base_key
        with open(base_path) as f:
            base_cfg = yaml.safe_load(f) or {}
        base_cfg.pop("_base_", None)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


# ── Seed setting ──────────────────────────────────────────────────────────────

def set_seed(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logger.info("Seeds set to %d", seed)


# ── Model factory ─────────────────────────────────────────────────────────────

def build_model(
    cfg: dict,
    model_name: Optional[str] = None,
) -> "HFModel":  # type: ignore[name-defined]
    """Build an HFModel from configuration.

    Args:
        cfg: Merged configuration dictionary.
        model_name: Optional model name override.

    Returns:
        Loaded :class:`~cocoa_cot.models.HFModel`.
    """
    from cocoa_cot.models.hf_model import HFModel
    from cocoa_cot.parsing.chain_parser import ChainParser

    model_cfg = cfg.get("model", {})
    name = model_name or model_cfg.get("name", "deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    device = model_cfg.get("device", "cuda")
    dtype = model_cfg.get("dtype", "bfloat16")
    max_new_tokens = cfg.get("sampling", {}).get("max_new_tokens", 512)
    cache_dir = cfg.get("cache", {}).get("dir", "results/cache")

    parser = ChainParser()
    return HFModel(
        model_name=name,
        device=device,
        dtype=dtype,
        parser=parser,
        max_new_tokens=max_new_tokens,
        cache_dir=cache_dir if cfg.get("cache", {}).get("enabled", True) else None,
    )


# ── Similarity factory ────────────────────────────────────────────────────────

def build_similarities(cfg: dict) -> tuple:
    """Build answer and step similarity functions from configuration.

    Args:
        cfg: Merged configuration dictionary.

    Returns:
        Tuple of ``(answer_sim, step_sim, nli_sim)``.
    """
    from cocoa_cot.similarity.cross_encoder import CrossEncoderSimilarity
    from cocoa_cot.similarity.nli import NLISimilarity
    from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity
    from cocoa_cot.parsing.step_segmenter import StepSegmenter

    sim_cfg = cfg.get("similarity", {})
    ce_model = sim_cfg.get("cross_encoder_model", "cross-encoder/stsb-roberta-large")
    nli_model = sim_cfg.get("nli_model", "cross-encoder/nli-deberta-v3-large")
    batch_size = sim_cfg.get("batch_size", 32)

    answer_sim = CrossEncoderSimilarity(model_name=ce_model, batch_size=batch_size)
    nli_sim = NLISimilarity(model_name=nli_model, batch_size=batch_size)
    segmenter = StepSegmenter()
    step_sim = StepAlignedSimilarity(answer_sim, segmenter)

    return answer_sim, step_sim, nli_sim


# ── CoCoA-CoT estimator factory ───────────────────────────────────────────────

def build_cocoa_cot(
    cfg: dict,
    model: "HFModel",  # type: ignore[name-defined]
    answer_sim,
    step_sim,
    confidence_type: Optional[str] = None,
    alpha: Optional[float] = None,
) -> "CoCoACoT":  # type: ignore[name-defined]
    """Build a CoCoACoT estimator from configuration.

    Args:
        cfg: Merged configuration dictionary.
        model: White-box HF model.
        answer_sim: Answer-level similarity function.
        step_sim: Step-aligned similarity function.
        confidence_type: Override confidence type (``"msp"``, ``"ppl"``, or ``"mte"``).
        alpha: Override alpha mixing parameter.

    Returns:
        :class:`~cocoa_cot.uncertainty.CoCoACoT` instance.
    """
    from cocoa_cot.parsing.chain_parser import ChainParser
    from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

    cot_cfg = cfg.get("cocoa_cot", {})
    sampling_cfg = cfg.get("sampling", {})

    _alpha = alpha if alpha is not None else cot_cfg.get("alpha", 0.5)
    _conf_type = confidence_type or cot_cfg.get("answer_confidence", "ppl")
    M = sampling_cfg.get("M", 10)
    temperature = sampling_cfg.get("temperature", 1.0)
    top_k = sampling_cfg.get("top_k", 50)
    top_p = sampling_cfg.get("top_p", 1.0)

    parser = model.parser

    return CoCoACoT(
        model=model,
        answer_similarity=answer_sim,
        step_similarity=step_sim,
        parser=parser,
        alpha=_alpha,
        M=M,
        confidence_type=_conf_type,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )


# ── Generation cache ──────────────────────────────────────────────────────────

def cache_generations(
    prompts: list[str],
    model: "HFModel",  # type: ignore[name-defined]
    M: int,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    cache_dir: str = "results/cache",
) -> list[dict]:
    """Pre-generate and cache greedy + M samples for all prompts.

    All methods (CoCoA variants, SAR, SemanticEntropy) reuse the same cached
    generations to ensure fair comparison.

    Args:
        prompts: List of input prompts.
        model: White-box HF model.
        M: Number of samples.
        temperature: Sampling temperature.
        top_k: Top-k parameter.
        top_p: Nucleus parameter.
        cache_dir: Cache directory.

    Returns:
        List of generation dicts, one per prompt, each with:
            - ``"greedy"``: GenerationOutput
            - ``"samples"``: list[GenerationOutput]
    """
    from tqdm import tqdm

    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    key_content = f"{model.model_name}|M{M}|t{temperature}|k{top_k}|p{top_p}|{len(prompts)}"
    cache_key = hashlib.sha256(key_content.encode()).hexdigest()[:16]
    cache_file = cache_path / f"generations_{cache_key}.pkl"

    if cache_file.exists():
        logger.info("Loading cached generations from %s", cache_file)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logger.info("Generating outputs for %d prompts (M=%d)…", len(prompts), M)
    results = []
    for prompt in tqdm(prompts, desc="Generating"):
        greedy = model.generate_greedy(prompt)
        samples = model.generate_sample(
            prompt, M=M, temperature=temperature, top_k=top_k, top_p=top_p
        )
        results.append({"greedy": greedy, "samples": samples, "prompt": prompt})

    with open(cache_file, "wb") as f:
        pickle.dump(results, f)
    logger.info("Generations cached to %s", cache_file)

    return results


# ── Result saving ─────────────────────────────────────────────────────────────

def save_results(
    records: list[dict],
    output_path: str,
) -> None:
    """Save result records to CSV.

    Args:
        records: List of result dicts.
        output_path: Path to output CSV file.
    """
    import pandas as pd

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    logger.info("Results saved to %s", output_path)


def print_rich_table(df: "pd.DataFrame", title: str = "Results") -> None:  # type: ignore[name-defined]
    """Print a rich-formatted results table to the terminal.

    Args:
        df: Results DataFrame.
        title: Table title.
    """
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title=title, show_header=True, header_style="bold cyan")

        for col in df.columns:
            table.add_column(str(col), style="dim")

        for _, row in df.iterrows():
            table.add_row(*[f"{v:.4f}" if isinstance(v, float) else str(v) for v in row])

        console.print(table)
    except ImportError:
        print(df.to_string())
