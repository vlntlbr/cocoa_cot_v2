"""
Alpha sensitivity analysis (Table 3).

Grid search over alpha ∈ [0.1, 0.2, ..., 0.9] for CoCoA-CoT-PPL
across all 6 datasets.  Produces a PRR heatmap.

Usage:
    python -m cocoa_cot.experiments.run_alpha \\
        --config configs/base.yaml \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/alpha_sensitivity.csv
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

ALPHA_GRID = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml"),
    datasets: list[str] = typer.Option(ALL_DATASETS),
    model: Optional[str] = typer.Option(None),
    output: str = typer.Option("results/tables/alpha_sensitivity.csv"),
    seeds: list[int] = typer.Option([42, 123, 456]),
    n_eval: int = typer.Option(500),
    alpha_grid: list[float] = typer.Option(ALPHA_GRID),
    figure_path: str = typer.Option("results/figures/alpha_heatmap.png"),
) -> None:
    """Alpha sensitivity analysis (Table 3)."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval

    console.rule("[bold blue]Alpha Sensitivity Analysis")

    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, _ = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]{dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits
        from cocoa_cot.evaluation.quality import get_quality_fn
        from cocoa_cot.evaluation.metrics import prr
        from cocoa_cot.uncertainty.information import PPLEstimator
        from cocoa_cot.uncertainty.consistency import ConsistencyEstimator

        for seed in seeds:
            set_seed(seed)
            splits = load_dataset_splits(dataset_name, n_eval=n_eval, seed=seed)
            eval_data = splits["eval"]
            prompts = [r["prompt"] for r in eval_data]
            gold_answers = [r["answer"] for r in eval_data]

            sampling_cfg = cfg.get("sampling", {})
            generations = cache_generations(
                prompts, hf_model,
                M=sampling_cfg.get("M", 10),
                temperature=sampling_cfg.get("temperature", 1.0),
                top_k=sampling_cfg.get("top_k", 50),
                top_p=sampling_cfg.get("top_p", 1.0),
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache"),
            )

            quality_fn = get_quality_fn(dataset_name)
            quality_scores = np.array([
                quality_fn(gen["greedy"].answer_text, gold)
                for gen, gold in zip(generations, gold_answers)
            ])

            # Pre-compute u_A, u_R, u_cons_A for all examples (constant across alpha)
            ppl_est = PPLEstimator(cot_mode=True)
            cons_est = ConsistencyEstimator(answer_sim)

            u_A_scores = []
            u_R_scores = []
            u_cons_scores = []

            for gen in generations:
                greedy = gen["greedy"]
                samples = gen["samples"]
                u_A_scores.append(ppl_est.estimate(greedy))
                u_R_raw = step_sim.compute_batch(
                    greedy.chain_text, [s.chain_text for s in samples]
                )
                u_R_scores.append(1.0 - float(np.mean(u_R_raw)))
                u_cons_scores.append(
                    cons_est.estimate(greedy.answer_text, [s.answer_text for s in samples])
                )

            u_A_arr = np.array(u_A_scores)
            u_R_arr = np.array(u_R_scores)
            u_cons_arr = np.array(u_cons_scores)

            for alpha in alpha_grid:
                unc_scores = (alpha * u_A_arr + (1.0 - alpha) * u_R_arr) * u_cons_arr
                prr_val = prr(unc_scores, quality_scores)

                all_records.append({
                    "alpha": alpha,
                    "dataset": dataset_name,
                    "seed": seed,
                    "prr": prr_val,
                })
                console.print(f"  alpha={alpha:.1f}: PRR={prr_val:.4f}")

    df = pd.DataFrame(all_records)
    save_results(all_records, output)

    # ── Summary table ─────────────────────────────────────────────────────────
    summary = (
        df.groupby(["alpha", "dataset"])["prr"]
        .mean()
        .round(4)
        .unstack("dataset")
        .reset_index()
    )
    console.print("\n[bold]PRR by alpha (mean over seeds):")
    console.print(summary.to_string(index=False))

    # Find best alpha per dataset
    for col in summary.columns[1:]:
        best_alpha = summary.loc[summary[col].idxmax(), "alpha"]
        console.print(f"  Best alpha for {col}: {best_alpha}")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    _plot_alpha_heatmap(df, figure_path)
    console.print(f"\n[green]Results saved to {output}")


def _plot_alpha_heatmap(df: pd.DataFrame, figure_path: str) -> None:
    """Plot and save alpha sensitivity heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from pathlib import Path

        Path(figure_path).parent.mkdir(parents=True, exist_ok=True)

        pivot = (
            df.groupby(["alpha", "dataset"])["prr"]
            .mean()
            .unstack("dataset")
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".3f",
            cmap="YlOrRd",
            ax=ax,
            vmin=0,
            vmax=1,
        )
        ax.set_title("PRR vs. Alpha (CoCoA-CoT-PPL)")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Alpha (α)")
        plt.tight_layout()
        plt.savefig(figure_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Alpha heatmap saved to %s", figure_path)
    except Exception as exc:
        logger.warning("Could not generate heatmap: %s", exc)


if __name__ == "__main__":
    app()
