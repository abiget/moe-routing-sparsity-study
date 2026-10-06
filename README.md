# MoE Routing Sparsity Study

This repository contains code and experiments related to routing sparsity in Mixture of Experts (MoE) models. The goal is to analyze how the number of experts and routing sparsity affect model performance across different models and benchmarks.

## Diagnostics

`src/diagnostics.py` runs counterfactual diagnostics on MoE models. It can replace individual layers, compute KL-divergence measurements, and save the results.

The reported values are interpreted as follows:

- `KL(0)`: damage from skipping the whole MoE layer.
- `KL(1)`: damage from keeping only the router's top-1 expert.
- `KL(2)`: reconstruction error from the top-2 implementation.
- `delta_1 = KL(0) - KL(1)`: value recovered by the first expert.
- `delta_2 = KL(1) - KL(2)`: additional value recovered by the second expert.

The study looks for variation across tokens, tokens where the second expert adds little value, tokens where one expert is already valuable, and tokens where the whole MoE layer can potentially be skipped.

To run the diagnostics, use the following command:

```bash
python -m scripts/run_diagnostics --model <model_name> --cache_dir <cache_dir> --text_file <text_file>

example:
    python scripts/run_diagnostics.py \
    --model_name FlameF0X/TinyMoE-100m-2x8-retrained \
    --cache_dir ./checkpoints \
    --text_file ./data/smollm_corpus_sample.txt
```