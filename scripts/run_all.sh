#!/usr/bin/env bash
# =============================================================================
# scripts/run_all.sh
#
# Runs the complete CoCoA-CoT experimental pipeline (all tables and figures).
# Usage:
#   bash scripts/run_all.sh [--model <model_name>] [--config <config_path>]
#
# Steps:
#   1. Table 1 — main results across all datasets and methods
#   2. Table 2 — ablation study
#   3. Table 3 — alpha sensitivity analysis
#   4. Calibration analysis with reliability diagrams
#   5. Black-box experiment
#   6. CoCoA-CoT Light training and evaluation
# =============================================================================

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
MODEL="${MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-8B}"
CONFIG="${CONFIG:-configs/base.yaml}"
SEEDS="${SEEDS:-42 123 456}"
N_EVAL="${N_EVAL:-500}"
N_HOLDOUT="${N_HOLDOUT:-2000}"
OUTPUT_DIR="${OUTPUT_DIR:-results}"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case $1 in
    --model)    MODEL="$2";    shift 2 ;;
    --config)   CONFIG="$2";   shift 2 ;;
    --n-eval)   N_EVAL="$2";   shift 2 ;;
    --output)   OUTPUT_DIR="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# ── Ensure results dirs exist ─────────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/tables" "${OUTPUT_DIR}/figures" "${OUTPUT_DIR}/cache"

echo "======================================================================"
echo "  CoCoA-CoT Full Experimental Pipeline"
echo "  Model  : ${MODEL}"
echo "  Config : ${CONFIG}"
echo "  N_eval : ${N_EVAL}"
echo "  Output : ${OUTPUT_DIR}"
echo "======================================================================"
date

# ── Step 1: Main results (Table 1) ───────────────────────────────────────────
echo ""
echo "[Step 1/6] Main results — Table 1"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_main \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --output "${OUTPUT_DIR}/tables/main_results.csv"

# ── Step 2: Ablation study (Table 2) ─────────────────────────────────────────
echo ""
echo "[Step 2/6] Ablation study — Table 2"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_ablation \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --output "${OUTPUT_DIR}/tables/ablation.csv"

# ── Step 3: Alpha sensitivity (Table 3) ──────────────────────────────────────
echo ""
echo "[Step 3/6] Alpha sensitivity — Table 3"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_alpha \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --output "${OUTPUT_DIR}/tables/alpha_sensitivity.csv" \
  --figure-path "${OUTPUT_DIR}/figures/alpha_heatmap.png"

# ── Step 4: Calibration analysis ─────────────────────────────────────────────
echo ""
echo "[Step 4/6] Calibration analysis"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_calibration \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --n-holdout "${N_HOLDOUT}" \
  --output "${OUTPUT_DIR}/tables/calibration.csv" \
  --figure-dir "${OUTPUT_DIR}/figures/calibration"

# ── Step 5: Black-box experiment ─────────────────────────────────────────────
echo ""
echo "[Step 5/6] Black-box experiment"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_blackbox \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --output "${OUTPUT_DIR}/tables/blackbox.csv"

# ── Step 6: CoCoA-CoT Light ───────────────────────────────────────────────────
echo ""
echo "[Step 6/6] CoCoA-CoT Light"
echo "----------------------------------------------------------------------"
python -m cocoa_cot.experiments.run_light \
  --config "${CONFIG}" \
  --model "${MODEL}" \
  --n-eval "${N_EVAL}" \
  --n-holdout "${N_HOLDOUT}" \
  --output "${OUTPUT_DIR}/tables/light.csv" \
  --model-save-path "${OUTPUT_DIR}/cache/aux_model.pt" \
  --figure-dir "${OUTPUT_DIR}/figures/light"

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo "  All experiments complete!"
echo "  Results saved to: ${OUTPUT_DIR}"
echo "======================================================================"
date
