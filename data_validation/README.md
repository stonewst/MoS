# Data Validation

This module turns a raw image-text dataset into a quality-scored one, in two
steps:

```
data_validation/
├── compute_baseline_score/   # Step 1: score each pair with several VLMs (CLIP, BLIP, ...)
└── compute_mos/              # Step 2: ensemble the scores into the robust MoS score
```

## Pipeline

```
              compute_baseline_score                        compute_mos
raw data_info  ────────────────────────►  data_info_all_scores  ──────►  data_info_mos
 (filename,        CLIP / BLIP / ...        (S^1, ..., S^M per pair)      (+ mos_score)
  caption)         + merge_scores.py                                     [N, M] -> [N]
```

1. **[Step 1 — compute_baseline_score](compute_baseline_score)**
   Compute a set of baseline quality scores `{S^1, ..., S^M}` for every
   image-text pair using different scoring models, then merge them (aligned by
   `filename`) into one unified `data_info`.

2. **[Step 2 — compute_mos](compute_mos)**
   Ensemble the `M` baseline scores of each pair into a single robust
   `mos_score` via the data-adaptive Mixture-of-Scores strategy (only 3 lines of
   core code).

The resulting `mos_score` then drives the downstream data-processing strategies
before vision-language pre-training (see [`../training`](../training)).

## Requirements

- `torch`, `torchvision`, `Pillow`, `tqdm`
- [OpenAI CLIP](https://github.com/openai/CLIP) — for CLIP-family scores
- [open_clip](https://github.com/mlfoundations/open_clip) — for EVA-CLIP-family scores
- [BLIP](https://github.com/salesforce/BLIP) — for BLIP-family scores
- [LAVIS](https://github.com/salesforce/LAVIS) — for BLIP2-family scores

Each sub-directory has its own `README.md` with detailed usage.
