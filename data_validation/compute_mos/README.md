# Step 2: Mixture-of-Scores (MoS)

Given the multiple baseline quality scores produced in
[Step 1](../compute_baseline_score), MoS integrates them into a single **robust**
score through a *data-adaptive ensemble*. The weight of each baseline score is
determined by two factors:

1. **Consensus** — the *density* of a score, i.e. its (negative) average
   distance to the other scores (Eq. (4)). Scores that most models agree on get
   larger weights.
2. **Quality uncertainty** — the *std* of a data point's scores, which sets a
   per-sample softmax *temperature* (Eq. (6)). Data with higher uncertainty get
   a smoother weight distribution so that more scoring models are considered.

The MoS score is the softmax-weighted sum of the baseline scores (Eq. (5), (7)).

## The 3-line core

The whole ensemble is only three lines (`mixture_of_scores` in
[`mos_core.py`](mos_core.py), matching Algorithm 1 of the paper):

```python
density = -(s[:, :, None] - s[:, None, :]).abs().sum(-1) / (M - 1)     # consensus,   [N, M]
temp    = (tmax - tmin) / (std.max() - std.min()) * std + (tmin * std.max() - tmax * std.min()) / (std.max() - std.min())  # uncertainty, [N]
mos     = (torch.softmax(density / temp[:, None], dim=-1) * s).sum(-1) # ensemble,    [N]
```

where `s` is the `[N, M]` matrix of `M` baseline scores for `N` samples.

## Usage

```bash
python compute_mos.py \
    --data_info_path /path/to/data_info_all_scores.json \
    --output_path    /path/to/data_info_mos.json \
    --score_keys clip-r50_similarity clip-vit-b32_similarity \
                 clip-vit-l14_similarity blip-large_itm-score \
    --normalize minmax --temp_min 0.5 --temp_max 1.5 \
    --save_key mos_score
```

`--auto_keys` can be used instead of `--score_keys` to ensemble every baseline
score found in the file. See [`run_mos.sh`](run_mos.sh) for a full example.

The output is the same `data_info` list with an extra `mos_score` field per
data point, ready to drive any downstream data-processing strategy (filtering,
sample weighting, re-captioning) before vision-language pre-training.

## Tips (from the paper's discussion)

- **More is not always better.** Ensembling fewer but critical scores can beat
  ensembling all of them (Table 3). MoS is nonetheless robust to the choice.
- If the per-score performance preference is known, ensemble the top-performing
  scores; otherwise pick scores from diverse architectures / model sizes to
  maximize complementarity.
- **Normalization.** Because different models output scores on different scales,
  per-score `minmax` normalization is applied by default so that the density and
  std are computed on a comparable scale.
- **Temperature bounds.** `temp_min=0.5`, `temp_max=1.5` are the paper's
  defaults; `L1` distance is used for density.
