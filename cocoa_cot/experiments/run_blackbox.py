"""
Black-box experiment.

Simulates a black-box API setting where logits and hidden states are
unavailable.  Evaluates Û_BB = Û_cons_A against full CoCoA-CoT-PPL
and reports the PRR gap per dataset.

Usage:
    python -m cocoa_cot.experiments.run_blackbox \\
        --config configs/base.yaml \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/blackbox.csv
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

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


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml"),
    datasets: list[str] = typer.Option(ALL_DATASETS),
    model: Optional[str] = typer.Option(None),
    output: str = typer.Option("results/tables/blackbox.csv"),
    seeds: list[int] = typer.Option([42, 123, 456]),
    n_eval: int = typer.Option(500),
) -> None:
    """Black-box vs white-box uncertainty quantification comparison."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval

    console.rule("[bold blue]Black-Box Experiment")

    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, _ = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]{dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits
        from cocoa_cot.evaluation.quality import get_quality_fn
        from cocoa_cot.evaluation.metrics import prr, auroc

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

            alpha = cfg.get("cocoa_cot", {}).get("alpha", 0.5)

            full_scores, bb_scores = _compute_scores(
                generations, answer_sim, step_sim, alpha
            )

            prr_full = prr(full_scores, quality_scores)
            prr_bb = prr(bb_scores, quality_scores)
            prr_gap = prr_full - prr_bb

            auroc_full = auroc(full_scores, quality_scores)
            auroc_bb = auroc(bb_scores, quality_scores)

            record = {
                "dataset": dataset_name,
                "seed": seed,
                "prr_full": round(prr_full, 4),
                "prr_blackbox": round(prr_bb, 4),
                "prr_gap": round(prr_gap, 4),
                "auroc_full": round(auroc_full, 4),
                "auroc_blackbox": round(auroc_bb, 4),
            }
            all_records.append(record)
            console.print(
                f"  PRR full={prr_full:.4f}, BB={prr_bb:.4f}, gap={prr_gap:.4f}"
            )

    df = pd.DataFrame(all_records)
    save_results(all_records, output)

    # ── Print summary table ───────────────────────────────────────────────────
    summary = (
        df.groupby("dataset")[["prr_full", "prr_blackbox", "prr_gap"]]
        .mean()
        .round(4)
        .reset_index()
    )

    tbl = Table(title="Black-Box vs Full CoCoA-CoT (mean PRR over seeds)")
    for col in summary.columns:
        tbl.add_column(col, justify="right")
    for _, row in summary.iterrows():
        tbl.add_row(*[str(v) for v in row.values])
    console.print(tbl)

    # Mean PRR gap across datasets
    mean_gap = df["prr_gap"].mean()
    console.print(f"\n[bold]Mean PRR gap (full - black-box): {mean_gap:.4f}")
    console.print(f"[green]Results saved to {output}")

    # ── Bar chart ─────────────────────────────────────────────────────────────
    _plot_gap_chart(summary)


def _compute_scores(
    generations,
    answer_sim,
    step_sim,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        full_scores   — CoCoA-CoT-PPL scores using logits + chain
        bb_scores     — Black-box scores (Û_BB = Û_cons_A only)
    """
    from cocoa_cot.uncertainty.information import PPLEstimator
    from cocoa_cot.uncertainty.consistency import ConsistencyEstimator

    ppl_est = PPLEstimator(cot_mode=True)
    cons_est = ConsistencyEstimator(answer_sim)

    full_list = []
    bb_list = []

    for gen in generations:
        greedy = gen["greedy"]
        samples = gen["samples"]
        sample_answers = [s.answer_text for s in samples]

        # Black-box: only answer consistency
        u_cons = cons_est.estimate(greedy.answer_text, sample_answers)
        bb_list.append(u_cons)

        # Full CoCoA-CoT-PPL
        u_A = ppl_est.estimate(greedy)
        u_R_vals = step_sim.compute_batch(
            greedy.chain_text, [s.chain_text for s in samples]
        )
        u_R = 1.0 - float(np.mean(u_R_vals))
        full_list.append((alpha * u_A + (1.0 - alpha) * u_R) * u_cons)

    return np.array(full_list), np.array(bb_list)


def _plot_gap_chart(summary: pd.DataFrame) -> None:
    """Bar chart showing PRR gap per dataset."""
    try:
        import matplotlib.pyplot as plt
        from pathlib import Path

        fig_path = "results/figures/blackbox_gap.png"
        Path(fig_path).parent.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(8, 5))
        x = range(len(summary))
        width = 0.35
        ax.bar(
            [i - width / 2 for i in x],
            summary["prr_full"],
            width,
            label="Full CoCoA-CoT",
            color="steelblue",
        )
        ax.bar(
            [i + width / 2 for i in x],
            summary["prr_blackbox"],
            width,
            label="Black-box",
            color="darkorange",
        )
        ax.set_xticks(list(x))
        ax.set_xticklabels(summary["dataset"], rotation=15)
        ax.set_ylabel("PRR")
        ax.set_ylim(0, 1)
        ax.set_title("Full vs Black-Box CoCoA-CoT: PRR by Dataset")
        ax.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Bar chart saved to %s", fig_path)
    except Exception as exc:
        logger.warning("Could not generate bar chart: %s", exc)


if __name__ == "__main__":
    app()
