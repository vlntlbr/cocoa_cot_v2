#!/usr/bin/env bash
# =============================================================================
# scripts/download_models.sh
#
# Downloads the models required by the CoCoA-CoT pipeline:
#   1. DeepSeek-R1-Distill-Llama-8B  (main reasoning model)
#   2. cross-encoder/stsb-roberta-large  (answer + step similarity)
#   3. cross-encoder/nli-deberta-v3-large  (NLI / SemanticEntropy)
#
# Requirements:
#   pip install huggingface_hub
#   (optional) huggingface-cli login  — for gated models
#
# Usage:
#   bash scripts/download_models.sh [--cache-dir <path>]
# =============================================================================

set -euo pipefail

CACHE_DIR="${HF_HOME:-${HOME}/.cache/huggingface/hub}"

while [[ $# -gt 0 ]]; do
  case $1 in
    --cache-dir) CACHE_DIR="$2"; shift 2 ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

export HF_HOME="${CACHE_DIR}"
echo "HuggingFace cache dir: ${CACHE_DIR}"
mkdir -p "${CACHE_DIR}"

# ── Helper: download a model repo snapshot ────────────────────────────────────
download_model() {
  local repo_id="$1"
  local description="$2"
  echo ""
  echo "----------------------------------------------------------------------"
  echo "  Downloading: ${repo_id}"
  echo "  Description: ${description}"
  echo "----------------------------------------------------------------------"
  python - <<PYEOF
from huggingface_hub import snapshot_download
import os
path = snapshot_download(
    repo_id="${repo_id}",
    cache_dir="${CACHE_DIR}",
    ignore_patterns=["*.msgpack", "*.h5", "flax_model*"],
)
print(f"Downloaded to: {path}")
PYEOF
}

echo "======================================================================"
echo "  CoCoA-CoT: Model Download"
echo "======================================================================"

# 1. Main reasoning model
download_model \
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  "DeepSeek-R1 distilled Llama 8B — primary reasoning model"

# 2. Cross-encoder for answer + step similarity
download_model \
  "cross-encoder/stsb-roberta-large" \
  "STS-B RoBERTa-Large cross-encoder — answer and step similarity"

# 3. NLI model for SemanticEntropy / bidirectional entailment clustering
download_model \
  "cross-encoder/nli-deberta-v3-large" \
  "NLI DeBERTa-v3-Large cross-encoder — NLI similarity / SemanticEntropy"

echo ""
echo "======================================================================"
echo "  All models downloaded successfully."
echo "  Cache location: ${CACHE_DIR}"
echo "======================================================================"

# Print total disk usage
du -sh "${CACHE_DIR}" 2>/dev/null || true
