# Step 1: Get Baseline Quality Scores

This step computes a set of **baseline quality scores** for every image-text
pair using several off-the-shelf vision-language models. Following the paper, we
cover four scoring-model families — **CLIP, EVA-CLIP, BLIP, BLIP2** — with one
script each:

| Script | Model family (paper Table 2) | Backend | Score(s) |
|--------|------------------------------|---------|----------|
| `compute_clip_score.py`    | CLIP (R50, ViT-B/32, ViT-L/14) | OpenAI `clip` | cosine similarity |
| `compute_evaclip_score.py` | EVA-CLIP (ViT-L/14, ViT-g/14)  | `open_clip`   | cosine similarity |
| `compute_blip_score.py`    | BLIP (ViT-B/16, ViT-L/16)      | official BLIP repo | ITM score + ITC similarity |
| `compute_blip2_score.py`   | BLIP2 (ViT-L/14, ViT-g/14)     | LAVIS         | ITM score + ITC similarity |

These scores are the raw material that the [MoS](../compute_mos) step later ensembles.

> **Environment:** all scripts run in the single `mos` env (see
> [`installation.md`](../../installation.md)). `numpy<2` and `transformers<4.27`
> are pinned for the whole pipeline; only **`timm`** must be swapped between the
> EVA-CLIP and BLIP2 scorers (commands are shown next to each usage below).

For a pair `(x_I, x_T)`, the CLIP / EVA-CLIP quality score is the cosine
similarity between the L2-normalized image embedding and text embedding
(Eq. (1)-(2) in the paper). BLIP / BLIP2 additionally offer an image-text
matching (ITM) probability.

## Data format

The dataset is a list of dicts saved as a `.json` file (`data_info`). Each dict
must at least contain:

```python
{
    "filename": "sub_dir/000000.jpg",   # relative to --image_root_path
    "caption":  "a photo of a cat",
    # ... other meta fields are kept untouched ...
}
```

Each scoring script writes the computed score back into every dict under a
`--save_key`, and re-saves the list.

## Usage

### 1) Compute per-model scores

CLIP-family scores (one cosine-similarity value per pair):

```bash
python compute_clip_score.py \
    --data_info_path /path/to/data_info.json \
    --image_root_path /path/to/images \
    --clip_model_name ViT-B/32 \
    --save_root_path /path/to/scores/clip-vit-b32 \
    --save_filename data_info_clip-vit-b32.json \
    --save_key clip-vit-b32_similarity \
    --batchsize 100
```

EVA-CLIP-family scores (cosine similarity, loaded via `open_clip`).
Before running, make sure `timm` is new enough (and keep `numpy<2`):

```bash
pip install "timm>=1.0.17" "numpy<2"
```

```bash
python compute_evaclip_score.py \
    --data_info_path /path/to/data_info.json \
    --image_root_path /path/to/images \
    --model_name EVA02-L-14 --pretrained merged2b_s4b_b131k \
    --save_root_path /path/to/scores/evaclip-vit-l14 \
    --save_filename data_info_evaclip-vit-l14.json \
    --save_key evaclip-vit-l14_similarity \
    --batchsize 32
```

BLIP-family scores (ITM score + ITC similarity per pair):

```bash
python compute_blip_score.py \
    --data_info_path /path/to/data_info.json \
    --image_root_path /path/to/images \
    --blip_code_path /path/to/BLIP \
    --blip_model_ckpt /path/to/model_large.pth \
    --vit_arch large --image_size 224 \
    --save_root_path /path/to/scores/blip-large \
    --save_filename data_info_blip-large.json \
    --save_key blip-large
```

BLIP2-family scores (ITM score + ITC similarity, loaded via LAVIS).
Before running, pin LAVIS-compatible versions (this swaps `timm` back):

```bash
pip install "timm==0.4.12" "numpy<2"
```

```bash
python compute_blip2_score.py \
    --data_info_path /path/to/data_info.json \
    --image_root_path /path/to/images \
    --name blip2_image_text_matching --model_type pretrain \
    --save_root_path /path/to/scores/blip2-vit-g14 \
    --save_filename data_info_blip2-vit-g14.json \
    --save_key blip2-vit-g14
```

See [`scripts/run_clip.sh`](scripts/run_clip.sh),
[`scripts/run_evaclip.sh`](scripts/run_evaclip.sh),
[`scripts/run_blip.sh`](scripts/run_blip.sh) and
[`scripts/run_blip2.sh`](scripts/run_blip2.sh) for multi-model examples.

> For large datasets, split the work with `--start_id` / `--end_id` and run the
> shards in parallel on multiple GPUs; then merge them with `concat` below.

### 2) Merge the scores

Concatenate the shards of a single score:

```bash
python merge_scores.py concat \
    --inputs /path/to/scores/clip-vit-b32/data_info_clip-vit-b32_*.json \
    --output /path/to/scores/data_info_clip-vit-b32.json \
    --check_unique
```

Join all baseline scores (aligned by `filename`) into one unified `data_info`,
which is exactly the input the MoS step expects:

```bash
python merge_scores.py join \
    --inputs /path/to/scores/data_info_clip-vit-b32.json \
             /path/to/scores/data_info_clip-vit-l14.json \
             /path/to/scores/data_info_blip-large.json \
    --output /path/to/scores/data_info_all_scores.json
```

## Notes

- **CLIP** uses the OpenAI `clip` package
  (`pip install git+https://github.com/openai/CLIP.git`).
- **EVA-CLIP** uses `open_clip` (`pip install open_clip_torch`); choose the
  `--model_name` / `--pretrained` pair from `open_clip.list_pretrained()`.
  Requires `timm>=1.0.17` (and `numpy<2`).
- **BLIP** relies on the official
  [BLIP repo](https://github.com/salesforce/BLIP); pass its path via
  `--blip_code_path` and the checkpoint via `--blip_model_ckpt`.
- **BLIP2** uses [LAVIS](https://github.com/salesforce/LAVIS)
  (`pip install salesforce-lavis`) with the `blip2_image_text_matching` model.
  Requires `timm==0.4.12`, `transformers<4.27` and `numpy<2`.
- Any dual-encoder VLM with an aligned image/text embedding space can serve as a
  scoring model; simply produce a `--save_key` field in the `data_info`.
