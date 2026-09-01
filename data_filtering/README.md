# Data Filtering

One of the data-processing strategies in the paper. Given a `data_info` with a
quality score (e.g. `mos_score` from [`../data_validation`](../data_validation)),
we sort the pairs by that score and keep the top **`keep_ratio`** fraction as the
training set; the lowest-quality pairs are removed.

## Usage

```bash
python filter_data.py \
    --data_info_path /path/to/data_info_mos.json \
    --output_path    /path/to/data_info_filtered.json \
    --score_key mos_score \
    --keep_ratio 0.8
```

- `--score_key`   which field to sort by (default `mos_score`; can be any single
  baseline score too, e.g. `clip-vit-b32_similarity`).
- `--keep_ratio`  fraction of highest-scoring data to keep, in `(0, 1]`
  (`0.8` = keep top 80%, remove lowest 20% — the paper's filtering ratio).
- `--ascending`   keep the lowest-scoring data instead (for analysis).
- `--removed_path` optionally dump the removed subset.

The output `data_info_filtered.json` has the same format as the input and can be
fed directly into [`../training`](../training).
