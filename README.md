# CoCoA-CoT

**CoCoA-CoT: Extending Confidence and Consistency-Based Uncertainty Quantification to Reasoning Language Models**

This repository implements the complete experimental pipeline for CoCoA-CoT, which extends the CoCoA UQ framework (Vashurin et al., 2025, arXiv:2502.04964) to Chain-of-Thought / reasoning language models.

## Overview

CoCoA-CoT decomposes uncertainty into three components:

| Component | Symbol | Description |
|-----------|--------|-------------|
| Answer confidence | `u_A` | Token-level NLL over answer tokens (MSP/PPL/MTE) |
| Reasoning coherence | `u_R` | Step-aligned dissimilarity between greedy and sampled chains |
| Answer consistency | `Û_cons_A` | Semantic dissimilarity between greedy and sampled answers |

Combined as:

```
Û_CoCoA-CoT = [α · u_A + (1−α) · u_R] · Û_cons_A
```

The key novelty is the **step-aligned similarity** function:

```
s_step(c, c') = (1/K) Σₖ max_{k'} s_sem(cₖ, c'_{k'})
```

which robustly aligns reasoning steps across chains of differing lengths.

## Repository Structure

```
cocoa_cot/
├── cocoa_cot/           # Main package
│   ├── models/          # LLM wrappers (HuggingFace + black-box)
│   ├── parsing/         # Chain-of-Thought output parsing
│   ├── similarity/      # Similarity functions (cross-encoder, NLI, step-aligned)
│   ├── uncertainty/     # UQ estimators (baselines + CoCoA-CoT)
│   ├── evaluation/      # PRR, AUROC, ECE metrics
│   ├── data/            # Dataset loaders for 6 benchmarks
│   ├── light/           # CoCoA-CoT Light (MLP auxiliary model)
│   └── experiments/     # Experiment scripts (Tables 1-3 of paper)
├── configs/             # YAML configuration files
├── scripts/             # Shell scripts for full pipeline
├── results/             # Auto-generated tables and figures
└── tests/               # Unit tests
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from cocoa_cot.models.hf_model import HFModel
from cocoa_cot.parsing.chain_parser import ChainParser
from cocoa_cot.uncertainty.cocoa_cot import CoCoACoT
from cocoa_cot.similarity.step_aligned import StepAlignedSimilarity
from cocoa_cot.similarity.cross_encoder import CrossEncoderSimilarity

parser = ChainParser()
sim = CrossEncoderSimilarity()
step_sim = StepAlignedSimilarity(sim, parser.segmenter)
model = HFModel("deepseek-ai/DeepSeek-R1-Distill-Llama-8B", "cuda", "bfloat16", parser)
estimator = CoCoACoT(model, sim, step_sim, parser, alpha=0.5, M=10, confidence_type="ppl")
result = estimator.estimate("What is 12 times 15? Think step by step.")
print(result)
```

## Reproducing Paper Results

```bash
# Full pipeline (~6 hours on A100)
bash scripts/run_all.sh

# Smoke test on 50 examples
python -m cocoa_cot.experiments.run_main \
    --config configs/base.yaml \
    --datasets gsm8k \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --n_eval 50 \
    --output results/tables/smoke_test.csv
```

## Datasets

| Dataset | Task | Quality Metric |
|---------|------|----------------|
| GSM8K | Math word problems | Exact numeric match |
| MATH-500 | Competition math | SymPy symbolic equality |
| HotPotQA | Multi-hop QA | AlignScore (cosine proxy) |
| ARC-Challenge | Science MCQ | Exact match |
| ProntoQA | Logical reasoning | Exact match |
| LiveCodeBench | Code generation | pass@1 (subprocess) |

## Methods Evaluated

- **Information-based**: MSP, PPL, MTE
- **Consistency-based**: DegMat, Û_cons
- **Hybrid**: SemanticEntropy, SAR
- **CoCoA baselines**: CoCoA-MSP, CoCoA-PPL, CoCoA-MTE
- **Proposed**: CoCoA-CoT-MSP, CoCoA-CoT-PPL, CoCoA-CoT-MTE, CoCoA-CoT-Light-PPL

## Tests

```bash
python -m pytest tests/ -v
```

## Citation

```bibtex
@article{vashurin2025cocoa,
  title={CoCoA: Confidence and Consistency-Based Approaches for Uncertainty Quantification in LLMs},
  author={Vashurin et al.},
  journal={arXiv preprint arXiv:2502.04964},
  year={2025}
}
```
