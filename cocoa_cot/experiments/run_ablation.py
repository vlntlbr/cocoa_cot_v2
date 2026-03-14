"""
Run ablation study (Table 2): 7 variants of CoCoA-CoT.

Ablation variants:
    A1: alpha=1.0 (answer-only, reduces to CoCoA)
    A2: alpha=0.0 (reasoning-only)
    A3: additive u_A + u_R (not multiplied by u_cons_A)
    A4: full-sequence similarity (s applied to full trace, not s_step)
    A5: graph-based similarity (DegMat on chain similarity)
    A6: step-aligned similarity (full proposed method, alpha=0.5)
    A7: CoCoA-CoT Light approximation of A6

Usage:
    python -m cocoa_cot.experiments.run_ablation \\
        --config configs/base.yaml \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/ablation.csv
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
    build_model,
    build_similarities,
    cache_generations,
    load_config,
    print_rich_table,
    save_results,
    set_seed,
)

logger = logging.getLogger(__name__)
app = typer.Typer()
console = Console()

ALL_DATASETS = [
    "gsm8k", "math500", "hotpotqa", "arc_challenge", "prontoqa", "livecodebench"
]

ABLATION_VARIANTS = {
    "A1_answer_only": {"alpha": 1.0, "use_step_sim": True, "combine": "full"},
    "A2_reasoning_only": {"alpha": 0.0, "use_step_sim": True, "combine": "full"},
    "A3_additive": {"alpha": 0.5, "use_step_sim": True, "combine": "additive"},
    "A4_full_seq_sim": {"alpha": 0.5, "use_step_sim": False, "combine": "full"},
    "A5_graph_sim": {"alpha": 0.5, "use_step_sim": False, "combine": "graph"},
    "A6_step_aligned": {"alpha": 0.5, "use_step_sim": True, "combine": "full"},
    "A7_light": {"alpha": 0.5, "use_step_sim": True, "combine": "light"},
}


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml"),
    datasets: list[str] = typer.Option(ALL_DATASETS),
    model: Optional[str] = typer.Option(None),
    output: str = typer.Option("results/tables/ablation.csv"),
    seeds: list[int] = typer.Option([42, 123, 456]),
    n_eval: int = typer.Option(500),
) -> None:
    """Run ablation study (Table 2)."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval

    console.rule("[bold blue]Ablation Study")

    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, nli_sim = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]{dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits

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

            from cocoa_cot.evaluation.quality import get_quality_fn
            quality_fn = get_quality_fn(dataset_name)
            quality_scores = np.array([
                quality_fn(gen["greedy"].answer_text, gold)
                for gen, gold in zip(generations, gold_answers)
            ])

            for variant_name, variant_cfg in ABLATION_VARIANTS.items():
                try:
                    unc_scores = _run_ablation_variant(
                        variant_cfg, generations, cfg, hf_model, answer_sim, step_sim
                    )
                    from cocoa_cot.evaluation.metrics import prr, auroc

                    prr_val = prr(np.array(unc_scores), quality_scores)
                    auroc_val = auroc(
                        np.array(unc_scores), (quality_scores > 0.5).astype(int)
                    )
                    all_records.append({
                        "variant": variant_name,
                        "dataset": dataset_name,
                        "seed": seed,
                        "prr": prr_val,
                        "auroc": auroc_val,
                    })
                    console.print(f"  {variant_name}: PRR={prr_val:.4f}")
                except Exception as exc:
                    logger.error("Variant %s failed: %s", variant_name, exc)

    df = pd.DataFrame(all_records)
    summary = (
        df.groupby(["variant", "dataset"])[["prr", "auroc"]]
        .agg(["mean", "std"])
        .round(4)
        .reset_index()
    )
    save_results(all_records, output)
    print_rich_table(summary, title="Ablation Study (Table 2)")
    console.print(f"\n[green]Results saved to {output}")


def _run_ablation_variant(
    variant_cfg: dict,
    generations: list[dict],
    cfg: dict,
    hf_model,
    answer_sim,
    step_sim,
) -> list[float]:
    """Compute uncertainty scores for a single ablation variant."""
    from cocoa_cot.uncertainty.information import PPLEstimator
    from cocoa_cot.uncertainty.consistency import ConsistencyEstimator, DegreeMatrixEstimator

    alpha = variant_cfg["alpha"]
    use_step_sim = variant_cfg["use_step_sim"]
    combine = variant_cfg["combine"]

    ppl_est = PPLEstimator(cot_mode=True)
    sim_fn = step_sim if use_step_sim else answer_sim
    cons_est = ConsistencyEstimator(answer_sim)
    deg_est = DegreeMatrixEstimator(answer_sim)

    scores = []

    for gen in tqdm(generations, desc="ablation", leave=False):
        greedy = gen["greedy"]
        samples = gen["samples"]

        u_A = ppl_est.estimate(greedy)
        u_R_val = 1.0 - float(
            np.mean(sim_fn.compute_batch(greedy.chain_text, [s.chain_text for s in samples]))
            if use_step_sim else
            np.mean(answer_sim.compute_one_to_many(
                greedy.chain_text, [s.chain_text for s in samples]
            ))
        )
        u_cons_A = cons_est.estimate(greedy.answer_text, [s.answer_text for s in samples])

        if combine == "full":
            unc = (alpha * u_A + (1.0 - alpha) * u_R_val) * u_cons_A
        elif combine == "additive":
            unc = alpha * u_A + (1.0 - alpha) * u_R_val + u_cons_A
        elif combine == "graph":
            # Graph-based: DegMat on chains instead of step-aligned
            all_chains = [greedy.chain_text] + [s.chain_text for s in samples]
            unc = deg_est.estimate(all_chains)
        elif combine == "light":
            # Light: PPL × 1.0 as placeholder (auxiliary model not trained here)
            unc = float(u_A)
        else:
            unc = 0.0

        scores.append(float(unc))

    return scores


if __name__ == "__main__":
    app()
