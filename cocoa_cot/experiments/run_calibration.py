"""
Calibration analysis.

Computes ECE (10 bins) for methods:
  [cocoa_cot_ppl, cocoa_ppl, sar, semantic_entropy]

Fits temperature scaling on holdout split; reports ECE before/after.
Saves reliability diagrams for each method × dataset combination.

Usage:
    python -m cocoa_cot.experiments.run_calibration \\
        --config configs/base.yaml \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/calibration.csv
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console

from cocoa_cot.experiments.utils import (
    build_model,
    build_similarities,
    cache_generations,
    load_config,
    save_results,
    set_seed,
)

logger = logging.getLogger(__name__)
app = typer.Typer()
console = Console()

ALL_DATASETS = [
    "gsm8k", "math500", "hotpotqa", "arc_challenge", "prontoqa", "livecodebench"
]

CALIB_METHODS = ["cocoa_cot_ppl", "cocoa_ppl", "sar", "semantic_entropy"]


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml"),
    datasets: list[str] = typer.Option(ALL_DATASETS),
    model: Optional[str] = typer.Option(None),
    output: str = typer.Option("results/tables/calibration.csv"),
    seeds: list[int] = typer.Option([42, 123, 456]),
    n_eval: int = typer.Option(500),
    n_holdout: int = typer.Option(2000),
    figure_dir: str = typer.Option("results/figures/calibration"),
    n_bins: int = typer.Option(10),
) -> None:
    """Calibration analysis with reliability diagrams and temperature scaling."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval
    cfg.setdefault("evaluation", {})["n_bins"] = n_bins

    console.rule("[bold blue]Calibration Analysis")

    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, nli_sim = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]{dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits
        from cocoa_cot.evaluation.quality import get_quality_fn
        from cocoa_cot.evaluation.metrics import ece, reliability_diagram
        from cocoa_cot.evaluation.calibration import temperature_scale

        for seed in seeds:
            set_seed(seed)

            splits = load_dataset_splits(
                dataset_name, n_eval=n_eval, n_holdout=n_holdout, seed=seed
            )
            eval_data = splits["eval"]
            holdout_data = splits["holdout"]

            quality_fn = get_quality_fn(dataset_name)
            sampling_cfg = cfg.get("sampling", {})

            # ── Eval split generations ────────────────────────────────────────
            eval_prompts = [r["prompt"] for r in eval_data]
            eval_gold = [r["answer"] for r in eval_data]
            eval_gens = cache_generations(
                eval_prompts, hf_model,
                M=sampling_cfg.get("M", 10),
                temperature=sampling_cfg.get("temperature", 1.0),
                top_k=sampling_cfg.get("top_k", 50),
                top_p=sampling_cfg.get("top_p", 1.0),
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache"),
            )
            eval_quality = np.array([
                quality_fn(gen["greedy"].answer_text, gold)
                for gen, gold in zip(eval_gens, eval_gold)
            ])

            # ── Holdout split generations (for temperature scaling) ───────────
            holdout_prompts = [r["prompt"] for r in holdout_data]
            holdout_gold = [r["answer"] for r in holdout_data]
            holdout_gens = cache_generations(
                holdout_prompts, hf_model,
                M=sampling_cfg.get("M", 10),
                temperature=sampling_cfg.get("temperature", 1.0),
                top_k=sampling_cfg.get("top_k", 50),
                top_p=sampling_cfg.get("top_p", 1.0),
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache"),
            )
            holdout_quality = np.array([
                quality_fn(gen["greedy"].answer_text, gold)
                for gen, gold in zip(holdout_gens, holdout_gold)
            ])

            # ── Compute uncertainty for each method ───────────────────────────
            method_scores = _compute_all_methods(
                eval_gens, holdout_gens, answer_sim, step_sim, nli_sim, cfg
            )

            for method, (eval_scores, holdout_scores) in method_scores.items():
                # ECE before temperature scaling
                ece_before = ece(eval_scores, eval_quality, n_bins=n_bins)

                # Fit temperature scaling on holdout
                try:
                    opt_temp, calib_eval_scores = temperature_scale(
                        holdout_scores, holdout_quality, eval_scores
                    )
                    ece_after = ece(calib_eval_scores, eval_quality, n_bins=n_bins)
                    logger.info("Optimal temperature for %s/%s: %.3f", method, dataset_name, opt_temp)
                except Exception as exc:
                    logger.warning("Temperature scaling failed for %s: %s", method, exc)
                    ece_after = ece_before
                    opt_temp = 1.0

                # Reliability diagram
                fig_path = f"{figure_dir}/{dataset_name}_{method}_seed{seed}.png"
                try:
                    reliability_diagram(
                        eval_scores,
                        eval_quality,
                        n_bins=n_bins,
                        title=f"{method} — {dataset_name} (seed={seed})",
                        save_path=fig_path,
                    )
                except Exception as exc:
                    logger.warning("Reliability diagram failed: %s", exc)

                record = {
                    "dataset": dataset_name,
                    "method": method,
                    "seed": seed,
                    "ece_before": round(ece_before, 4),
                    "ece_after": round(ece_after, 4),
                    "ece_reduction": round(ece_before - ece_after, 4),
                    "optimal_temperature": round(opt_temp, 4),
                }
                all_records.append(record)
                console.print(
                    f"  [{method}] ECE before={ece_before:.4f} after={ece_after:.4f} T*={opt_temp:.3f}"
                )

    df = pd.DataFrame(all_records)
    save_results(all_records, output)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary = (
        df.groupby(["method", "dataset"])[["ece_before", "ece_after"]]
        .mean()
        .round(4)
    )
    console.print("\n[bold]ECE summary (mean over seeds):")
    console.print(summary.to_string())
    console.print(f"\n[green]Results saved to {output}")


def _compute_all_methods(
    eval_gens,
    holdout_gens,
    answer_sim,
    step_sim,
    nli_sim,
    cfg: dict,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Compute uncertainty scores for eval and holdout splits.
    Returns dict: method -> (eval_scores, holdout_scores)
    """
    from cocoa_cot.uncertainty.information import PPLEstimator
    from cocoa_cot.uncertainty.consistency import ConsistencyEstimator
    from cocoa_cot.uncertainty.hybrid import SAREstimator, SemanticEntropyEstimator

    alpha = cfg.get("cocoa_cot", {}).get("alpha", 0.5)
    results: dict[str, tuple[list, list]] = {m: ([], []) for m in CALIB_METHODS}

    ppl_est = PPLEstimator(cot_mode=True)
    cons_est = ConsistencyEstimator(answer_sim)
    sar_est = SAREstimator(answer_sim)
    se_est = SemanticEntropyEstimator(nli_sim)

    def _score_split(generations):
        method_lists: dict[str, list] = {m: [] for m in CALIB_METHODS}
        for gen in generations:
            greedy = gen["greedy"]
            samples = gen["samples"]
            sample_texts = [s.answer_text for s in samples]

            # cocoa_cot_ppl
            u_A = ppl_est.estimate(greedy)
            u_R_vals = step_sim.compute_batch(greedy.chain_text, [s.chain_text for s in samples])
            u_R = 1.0 - float(np.mean(u_R_vals))
            u_cons = cons_est.estimate(greedy.answer_text, sample_texts)
            method_lists["cocoa_cot_ppl"].append(
                (alpha * u_A + (1.0 - alpha) * u_R) * u_cons
            )

            # cocoa_ppl (full-sequence)
            from cocoa_cot.uncertainty.information import PPLEstimator as FullPPL
            fppl = FullPPL(cot_mode=False)
            u_A_full = fppl.estimate(greedy)
            u_cons_full = cons_est.estimate(greedy.answer_text, sample_texts)
            method_lists["cocoa_ppl"].append(u_A_full * u_cons_full)

            # sar
            method_lists["sar"].append(sar_est.estimate(greedy, samples))

            # semantic_entropy
            all_texts = [greedy.answer_text] + sample_texts
            method_lists["semantic_entropy"].append(se_est.estimate(all_texts))

        return {k: np.array(v) for k, v in method_lists.items()}

    eval_scores = _score_split(eval_gens)
    holdout_scores = _score_split(holdout_gens)

    return {m: (eval_scores[m], holdout_scores[m]) for m in CALIB_METHODS}


if __name__ == "__main__":
    app()
