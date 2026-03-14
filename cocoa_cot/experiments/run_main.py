"""
Run main experiment (Table 1): all methods × all datasets.

Usage:
    python -m cocoa_cot.experiments.run_main \\
        --config configs/base.yaml \\
        --datasets gsm8k math500 hotpotqa arc_challenge prontoqa livecodebench \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/main_results.csv \\
        --seeds 42 123 456
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from tqdm import tqdm

from cocoa_cot.experiments.utils import (
    build_cocoa_cot,
    build_model,
    build_similarities,
    load_config,
    print_rich_table,
    save_results,
    set_seed,
)

logger = logging.getLogger(__name__)
app = typer.Typer()
console = Console()

METHODS = [
    "msp",
    "ppl",
    "mte",
    "degmat",
    "semantic_entropy",
    "sar",
    "cocoa_msp",
    "cocoa_ppl",
    "cocoa_mte",
    "cocoa_cot_msp",
    "cocoa_cot_ppl",
    "cocoa_cot_mte",
    "cocoa_cot_light_ppl",
]

ALL_DATASETS = [
    "gsm8k",
    # "math500",
    # "hotpotqa",
    # "arc_challenge",
    # "prontoqa",
    # "livecodebench",
]


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml", help="Path to config YAML"),
    datasets: list[str] = typer.Option(ALL_DATASETS, help="Datasets to evaluate"),
    model: Optional[str] = typer.Option(None, help="Model name override"),
    output: str = typer.Option("results/tables/main_results.csv", help="Output CSV path"),
    seeds: list[int] = typer.Option([42, 123, 456], help="Random seeds"),
    n_eval: int = typer.Option(500, help="Number of eval examples per dataset"),
    methods: list[str] = typer.Option(METHODS, help="Methods to evaluate"),
) -> None:
    """Run main results experiment (Table 1)."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)

    # Override n_eval in cfg
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval

    console.rule("[bold blue]CoCoA-CoT Main Experiment")
    console.print(f"Datasets: {datasets}")
    console.print(f"Methods:  {methods}")
    console.print(f"Seeds:    {seeds}")

    # ── Build shared components ───────────────────────────────────────────────
    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, nli_sim = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]Dataset: {dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits

        for seed in seeds:
            set_seed(seed)
            splits = load_dataset_splits(dataset_name, n_eval=n_eval, seed=seed)
            eval_data = splits["eval"]

            prompts = [r["prompt"] for r in eval_data]
            gold_answers = [r["answer"] for r in eval_data]

            # ── Pre-generate all samples once (cache for reuse) ───────────────
            from cocoa_cot.experiments.utils import cache_generations

            sampling_cfg = cfg.get("sampling", {})
            generations = cache_generations(
                prompts,
                hf_model,
                M=sampling_cfg.get("M", 10),
                temperature=sampling_cfg.get("temperature", 1.0),
                top_k=sampling_cfg.get("top_k", 50),
                top_p=sampling_cfg.get("top_p", 1.0),
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache"),
            )

            # ── Compute quality scores ────────────────────────────────────────
            from cocoa_cot.evaluation.quality import get_quality_fn

            quality_fn = get_quality_fn(dataset_name)
            quality_scores = np.array([
                quality_fn(
                    gen["greedy"].answer_text,
                    gold,
                    **({"test_cases": eval_data[i]["extra"].get("test_cases", [])}
                       if dataset_name == "livecodebench" else {})
                )
                for i, (gen, gold) in enumerate(zip(generations, gold_answers))
            ])

            # ── Run each method ───────────────────────────────────────────────
            for method_name in methods:
                try:
                    unc_scores = _run_method(
                        method_name=method_name,
                        generations=generations,
                        cfg=cfg,
                        hf_model=hf_model,
                        answer_sim=answer_sim,
                        step_sim=step_sim,
                        nli_sim=nli_sim,
                    )
                    from cocoa_cot.evaluation.metrics import prr, auroc

                    prr_val = prr(
                        np.array(unc_scores),
                        quality_scores,
                        rejection_max=cfg["evaluation"].get("rejection_max", 0.5),
                    )
                    auroc_val = auroc(
                        np.array(unc_scores),
                        (quality_scores > 0.5).astype(int),
                    )
                    record = {
                        "method": method_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "prr": prr_val,
                        "auroc": auroc_val,
                        "n_examples": len(eval_data),
                    }
                    all_records.append(record)
                    console.print(
                        f"  [{method_name}] PRR={prr_val:.4f}, AUROC={auroc_val:.4f}"
                    )
                except Exception as exc:
                    logger.error("Method %s failed on %s: %s", method_name, dataset_name, exc)

    # ── Aggregate over seeds ──────────────────────────────────────────────────
    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["method", "dataset"])[["prr", "auroc"]]
        .agg(["mean", "std"])
        .round(4)
    )
    summary.columns = ["prr_mean", "prr_std", "auroc_mean", "auroc_std"]
    summary = summary.reset_index()

    save_results(all_records, output)
    summary_path = output.replace(".csv", "_summary.csv")
    summary.to_csv(summary_path, index=False)

    print_rich_table(summary, title="Main Results (Table 1)")
    console.print(f"\n[green]Results saved to {output}")


def _run_method(
    method_name: str,
    generations: list[dict],
    cfg: dict,
    hf_model,
    answer_sim,
    step_sim,
    nli_sim,
) -> list[float]:
    """Compute uncertainty scores for all examples using a given method.

    Args:
        method_name: Name of the method to run.
        generations: Pre-computed generation outputs.
        cfg: Configuration dictionary.
        hf_model: White-box HF model.
        answer_sim: Answer similarity function.
        step_sim: Step-aligned similarity function.
        nli_sim: NLI similarity function (for SemanticEntropy).

    Returns:
        List of uncertainty scores, one per example.
    """
    from cocoa_cot.models.base import GenerationOutput
    from cocoa_cot.uncertainty.information import MSPEstimator, PPLEstimator, MTEEstimator
    from cocoa_cot.uncertainty.consistency import DegreeMatrixEstimator, ConsistencyEstimator
    from cocoa_cot.uncertainty.hybrid import SemanticEntropyEstimator, SAREstimator
    from cocoa_cot.uncertainty.cocoa import CoCoAEstimator
    from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT

    sampling_cfg = cfg.get("sampling", {})
    M = sampling_cfg.get("M", 10)

    scores = []

    if method_name in ("msp", "ppl", "mte"):
        conf_map = {"msp": MSPEstimator, "ppl": PPLEstimator, "mte": MTEEstimator}
        estimator = conf_map[method_name](cot_mode=True)
        for gen in tqdm(generations, desc=method_name, leave=False):
            scores.append(estimator.estimate(gen["greedy"]))

    elif method_name == "degmat":
        estimator = DegreeMatrixEstimator(answer_sim)
        for gen in tqdm(generations, desc=method_name, leave=False):
            answers = [s.answer_text for s in gen["samples"]] + [gen["greedy"].answer_text]
            scores.append(estimator.estimate(answers))

    elif method_name == "semantic_entropy":
        estimator = SemanticEntropyEstimator(nli_sim)
        for gen in tqdm(generations, desc=method_name, leave=False):
            answers = [s.answer_text for s in gen["samples"]]
            log_probs = [sum(s.token_logprobs) for s in gen["samples"]]
            scores.append(estimator.estimate(answers, log_probs))

    elif method_name == "sar":
        estimator = SAREstimator(answer_sim)
        for gen in tqdm(generations, desc=method_name, leave=False):
            candidates = [s.answer_text for s in gen["samples"]]
            scores.append(estimator.estimate(gen["greedy"], candidates))

    elif method_name in ("cocoa_msp", "cocoa_ppl", "cocoa_mte"):
        conf_type = method_name.split("_")[1]
        estimator = CoCoAEstimator(conf_type, answer_sim)
        for gen in tqdm(generations, desc=method_name, leave=False):
            scores.append(estimator.estimate(gen["greedy"], gen["samples"]))

    elif method_name in ("cocoa_cot_msp", "cocoa_cot_ppl", "cocoa_cot_mte"):
        conf_type = method_name.split("_")[-1]
        cot_cfg = cfg.get("cocoa_cot", {})
        alpha = cot_cfg.get("alpha", 0.5)

        estimator = CoCoACoT(
            model=hf_model,
            answer_similarity=answer_sim,
            step_similarity=step_sim,
            parser=hf_model.parser,
            alpha=alpha,
            M=M,
            confidence_type=conf_type,
            temperature=sampling_cfg.get("temperature", 1.0),
            top_k=sampling_cfg.get("top_k", 50),
            top_p=sampling_cfg.get("top_p", 1.0),
        )
        for gen in tqdm(generations, desc=method_name, leave=False):
            greedy = gen["greedy"]
            samples = gen["samples"]
            u_A = estimator._compute_u_A(greedy)
            u_R = estimator._compute_u_R(
                greedy.chain_text, [s.chain_text for s in samples]
            )
            u_cons_A = estimator._compute_u_cons_A(
                greedy.answer_text, [s.answer_text for s in samples]
            )
            unc = (alpha * u_A + (1.0 - alpha) * u_R) * u_cons_A
            scores.append(float(unc))

    elif method_name == "cocoa_cot_light_ppl":
        # CoCoA-CoT Light: requires trained auxiliary model
        # For main experiment, train on holdout split or skip if not available
        from cocoa_cot.light.aux_model import CoCoACoTLight

        light = CoCoACoTLight(
            layer_idx=cfg.get("model", {}).get("embedding_layer", 16),
            device=cfg.get("model", {}).get("device", "cuda"),
        )
        # Check for pre-trained model
        light_path = "results/cache/aux_model.pt"
        if not __import__("pathlib").Path(light_path).exists():
            logger.warning(
                "CoCoA-CoT Light model not found at %s. "
                "Run run_light.py first, or using PPL fallback.",
                light_path,
            )
            # Fallback to PPL
            estimator = PPLEstimator(cot_mode=True)
            for gen in tqdm(generations, desc=method_name, leave=False):
                scores.append(estimator.estimate(gen["greedy"]))
        else:
            light.load(light_path)
            for gen in tqdm(generations, desc=method_name, leave=False):
                scores.append(light.predict(gen["prompt"], hf_model))

    else:
        raise ValueError(f"Unknown method: {method_name!r}")

    return scores


if __name__ == "__main__":
    app()
