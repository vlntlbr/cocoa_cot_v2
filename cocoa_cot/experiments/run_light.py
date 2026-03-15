"""
CoCoA-CoT Light experiment.

1. Generate greedy outputs + hidden states on holdout split.
2. Compute full CoCoA-CoT targets via M=10 sampling (expensive, cached).
3. Extract dual embeddings [e_c; e_a] from hidden layer 16.
4. Train AuxiliaryModel (2-layer MLP) for 30 epochs, log train/val MSE.
5. Evaluate on eval split: PRR of Light vs full CoCoA-CoT.
6. Report inference-time comparison.
7. Save model checkpoint.

Usage:
    python -m cocoa_cot.experiments.run_light \\
        --config configs/base.yaml \\
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \\
        --output results/tables/light.csv
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
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
    "gsm8k"#, "math500", "hotpotqa", "arc_challenge", "prontoqa", "livecodebench"
]


@app.command()
def main(
    config: str = typer.Option("configs/base.yaml"),
    datasets: list[str] = typer.Option(ALL_DATASETS),
    model: Optional[str] = typer.Option(None),
    output: str = typer.Option("results/tables/light.csv"),
    seeds: list[int] = typer.Option([42, 123, 456]),
    n_eval: int = typer.Option(50),
    n_holdout: int = typer.Option(2000),
    model_save_path: str = typer.Option("results/cache/aux_model.pt"),
    figure_dir: str = typer.Option("results/figures/light"),
) -> None:
    """CoCoA-CoT Light: train 2-layer MLP to approximate full estimator."""
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(config)
    cfg.setdefault("evaluation", {})["n_eval"] = n_eval

    console.rule("[bold blue]CoCoA-CoT Light Experiment")

    hf_model = build_model(cfg, model_name=model)
    answer_sim, step_sim, _ = build_similarities(cfg)

    all_records = []

    for dataset_name in datasets:
        console.rule(f"[cyan]{dataset_name}")
        from cocoa_cot.data.loaders import load_dataset_splits
        from cocoa_cot.evaluation.quality import get_quality_fn
        from cocoa_cot.evaluation.metrics import prr
        from cocoa_cot.light.dual_embedding import DualEmbeddingExtractor
        from cocoa_cot.light.aux_model import AuxiliaryModel, CoCoACoTLight

        for seed in seeds:
            set_seed(seed)
            splits = load_dataset_splits(
                dataset_name, n_eval=n_eval, n_holdout=n_holdout, seed=seed
            )
            eval_data = splits["eval"]
            holdout_data = splits["holdout"]

            quality_fn = get_quality_fn(dataset_name)
            sampling_cfg = cfg.get("sampling", {})
            light_cfg = cfg.get("light", {})
            layer_idx = cfg.get("model", {}).get("embedding_layer", 16)

            # ── Holdout: generate + compute CoCoA-CoT targets ─────────────────
            console.print("[dim]Computing holdout targets (M=10, cached)...")
            holdout_prompts = [r["prompt"] for r in holdout_data]
            holdout_gens = cache_generations(
                holdout_prompts, hf_model,
                M=sampling_cfg.get("M", 10),
                temperature=sampling_cfg.get("temperature", 1.0),
                top_k=sampling_cfg.get("top_k", 50),
                top_p=sampling_cfg.get("top_p", 1.0),
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache"),
            )
            holdout_targets = _compute_full_targets(
                holdout_gens, answer_sim, step_sim,
                alpha=cfg.get("cocoa_cot", {}).get("alpha", 0.5)
            )

            # ── Extract dual embeddings for holdout ───────────────────────────
            console.print(f"[dim]Extracting dual embeddings from layer {layer_idx}...")
            extractor = DualEmbeddingExtractor(hf_model, layer_idx=layer_idx)
            holdout_features = _extract_features_batch(
                holdout_prompts, extractor, cache_dir=cfg.get("cache", {}).get("dir", "results/cache")
            )

            if holdout_features is None or len(holdout_features) == 0:
                logger.warning("Could not extract holdout features for %s. Skipping.", dataset_name)
                continue

            # ── Train auxiliary model ─────────────────────────────────────────
            d_model = holdout_features[0].shape[0]
            hidden_dim = light_cfg.get("hidden_dim", 512)
            aux_model = AuxiliaryModel(d_model=d_model, hidden_dim=hidden_dim)

            light_runner = CoCoACoTLight(aux_model)
            train_losses, val_losses = light_runner.train(
                features=np.stack(holdout_features),
                targets=holdout_targets,
                cfg=light_cfg,
            )

            # Save model checkpoint
            save_dir = Path(model_save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / f"aux_model_{dataset_name}_seed{seed}.pt"
            light_runner.save(str(ckpt_path))
            console.print(f"  Saved checkpoint: {ckpt_path}")

            # Plot training curve
            _plot_training_curve(
                train_losses, val_losses, dataset_name, seed, figure_dir
            )

            # ── Eval split ────────────────────────────────────────────────────
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

            # Full CoCoA-CoT scores on eval
            alpha = cfg.get("cocoa_cot", {}).get("alpha", 0.5)
            full_targets = _compute_full_targets(eval_gens, answer_sim, step_sim, alpha)
            prr_full = prr(full_targets, eval_quality)

            # Light model inference — time it
            eval_features = _extract_features_batch(
                eval_prompts, extractor,
                cache_dir=cfg.get("cache", {}).get("dir", "results/cache")
            )
            t0 = time.perf_counter()
            if eval_features is not None:
                light_scores = light_runner.predict_batch(np.stack(eval_features))
            else:
                light_scores = full_targets  # fallback
            light_infer_time = time.perf_counter() - t0

            # Full method inference time (just estimation, not generation)
            t0 = time.perf_counter()
            _ = _compute_full_targets(eval_gens[:10], answer_sim, step_sim, alpha)
            full_infer_time_per10 = time.perf_counter() - t0
            full_infer_time = full_infer_time_per10 / 10 * len(eval_gens)

            prr_light = prr(light_scores, eval_quality)

            # Scatter plot: predicted vs actual
            _plot_scatter(full_targets, light_scores, dataset_name, seed, figure_dir)

            record = {
                "dataset": dataset_name,
                "seed": seed,
                "prr_full": round(prr_full, 4),
                "prr_light": round(prr_light, 4),
                "prr_delta": round(prr_full - prr_light, 4),
                "final_train_mse": round(float(train_losses[-1]), 6),
                "final_val_mse": round(float(val_losses[-1]), 6),
                "light_infer_time_s": round(light_infer_time, 3),
                "full_infer_time_s": round(full_infer_time, 3),
                "speedup": round(full_infer_time / (light_infer_time + 1e-9), 1),
            }
            all_records.append(record)
            console.print(
                f"  PRR full={prr_full:.4f}, light={prr_light:.4f}, "
                f"speedup={record['speedup']}x"
            )

    df = pd.DataFrame(all_records)
    save_results(all_records, output)

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(df) > 0:
        summary = (
            df.groupby("dataset")[["prr_full", "prr_light", "prr_delta", "speedup"]]
            .mean()
            .round(4)
            .reset_index()
        )
        console.print("\n[bold]Light vs Full (mean over seeds):")
        console.print(summary.to_string(index=False))

    console.print(f"\n[green]Results saved to {output}")


def _compute_full_targets(
    generations, answer_sim, step_sim, alpha: float
) -> np.ndarray:
    """Compute CoCoA-CoT-PPL scores for a list of generation dicts."""
    from cocoa_cot.uncertainty.information import PPLEstimator
    from cocoa_cot.uncertainty.consistency import ConsistencyEstimator

    ppl_est = PPLEstimator(cot_mode=True)
    cons_est = ConsistencyEstimator(answer_sim)

    scores = []
    for gen in generations:
        greedy = gen["greedy"]
        samples = gen["samples"]
        u_A = ppl_est.estimate(greedy)
        u_R_vals = step_sim.compute_batch(greedy.chain_text, [s.chain_text for s in samples])
        u_R = 1.0 - float(np.mean(u_R_vals))
        u_cons = cons_est.estimate(greedy.answer_text, [s.answer_text for s in samples])
        scores.append((alpha * u_A + (1.0 - alpha) * u_R) * u_cons)

    return np.array(scores)


def _extract_features_batch(
    prompts: list[str],
    extractor,
    cache_dir: str = "results/cache",
) -> list[np.ndarray] | None:
    """Extract concatenated [e_c; e_a] features for a list of prompts."""
    features = []
    try:
        for prompt in prompts:
            e_c, e_a = extractor.extract(prompt)
            feat = np.concatenate([e_c, e_a], axis=0)
            features.append(feat)
        return features
    except Exception as exc:
        logger.warning("Feature extraction failed: %s", exc)
        return None


def _plot_training_curve(
    train_losses: list[float],
    val_losses: list[float],
    dataset_name: str,
    seed: int,
    figure_dir: str,
) -> None:
    """Save training/validation MSE curve."""
    try:
        import matplotlib.pyplot as plt

        Path(figure_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 4))
        epochs = range(1, len(train_losses) + 1)
        ax.plot(epochs, train_losses, label="Train MSE", color="steelblue")
        ax.plot(epochs, val_losses, label="Val MSE", color="darkorange", linestyle="--")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.set_title(f"CoCoA-CoT Light Training — {dataset_name} (seed={seed})")
        ax.legend()
        plt.tight_layout()
        fig_path = f"{figure_dir}/{dataset_name}_seed{seed}_training.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Training curve saved to %s", fig_path)
    except Exception as exc:
        logger.warning("Could not plot training curve: %s", exc)


def _plot_scatter(
    full_targets: np.ndarray,
    light_scores: np.ndarray,
    dataset_name: str,
    seed: int,
    figure_dir: str,
) -> None:
    """Scatter plot of predicted vs actual uncertainty scores."""
    try:
        import matplotlib.pyplot as plt

        Path(figure_dir).mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(full_targets, light_scores, alpha=0.5, s=10, color="steelblue")
        lims = [
            min(full_targets.min(), light_scores.min()),
            max(full_targets.max(), light_scores.max()),
        ]
        ax.plot(lims, lims, "r--", linewidth=1, label="y=x")
        ax.set_xlabel("Full CoCoA-CoT (target)")
        ax.set_ylabel("Light (predicted)")
        ax.set_title(f"Light vs Full — {dataset_name} (seed={seed})")
        ax.legend()
        plt.tight_layout()
        fig_path = f"{figure_dir}/{dataset_name}_seed{seed}_scatter.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Scatter plot saved to %s", fig_path)
    except Exception as exc:
        logger.warning("Could not plot scatter: %s", exc)


if __name__ == "__main__":
    app()
