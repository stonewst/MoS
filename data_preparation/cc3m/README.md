# CC3M Preparation

We use **Conceptual Captions 3M (CC3M)** as the running example. CC3M is a
dataset of ~3.3M images annotated with captions harvested from the web (from the
Alt-text HTML attribute of web images), covering a wide variety of styles. See
the [LAVIS dataset card](https://github.com/salesforce/LAVIS/blob/main/dataset_card/conceptual_captions.md)
for more background.

> **Note:** CC3M is distributed as a list of `(caption, url)` pairs, and the
> images are downloaded by requesting the URLs. Since some URLs disappear over
> time, the downloaded dataset is expected to be **partial** — this is normal.

The goal of this step is to produce a `data_info.json` file (a list of dicts)
that the [scoring step](../../data_validation/compute_baseline_score) consumes.

## Requirements

```bash
pip install img2dataset pandas pyarrow
```

## Step 1 — Get the URL list

Download `Train_GCC-training.tsv` (and optionally `Validation_GCC-1.1.0-Validation.tsv`)
from the [Conceptual Captions website](https://ai.google.com/research/ConceptualCaptions/download).
It is a tab-separated file with two columns — `caption` and `url` — and **no
header**.

## Step 2 — Download the images

```bash
bash download_cc3m.sh
```

This uses [img2dataset](https://github.com/rom1504/img2dataset) to fetch all
URLs. For each shard it writes a `{shard}/{key}.jpg` image folder together with a
`{shard}.parquet` metadata file. Edit `RAW_TSV` / `OUT_DIR` and the
`--processes_count` / `--thread_count` / `--image_size` options inside the script
to match your machine.

The output layout looks like:

```
cc3m_images/
├── 00000.parquet
├── 00000/
│   ├── 000000000.jpg
│   ├── 000000001.jpg
│   └── ...
├── 00001.parquet
├── 00001/
└── ...
```

## Step 3 — Build `data_info.json`

```bash
python build_data_info.py \
    --img2dataset_dir /path/to/cc3m_images \
    --output_path /path/to/cc3m/data_info.json
```

This scans every `{shard}.parquet`, keeps the successfully-downloaded pairs, and
saves a list of dicts. Each dict has the fields:

```python
{
    "filename": "00000/000000000.jpg",   # relative to the image root (cc3m_images/)
    "caption":  "a photo of ...",
    "url": ..., "key": ..., "status": "success",
    "width": ..., "height": ...,
    "original_width": ..., "original_height": ...,
    "md5": ..., "exif": ...,
}
```

## Next

Feed the resulting `data_info.json` (with `--image_root_path` pointing to
`cc3m_images/`) into the baseline-scoring step:

```bash
cd ../../data_validation/compute_baseline_score
python compute_clip_score.py \
    --data_info_path /path/to/cc3m/data_info.json \
    --image_root_path /path/to/cc3m_images \
    ...
```

Other datasets (CC12M, LAION, YFCC15M, ...) can be prepared the same way: the
only requirement is a `data_info.json` whose dicts contain `filename` (relative to
the image root) and `caption`.
