# Data Preparation

This module downloads an image-text dataset and converts it into the
`data_info.json` format consumed by the rest of the pipeline. A `data_info` is a
list of dicts, each describing one image-text pair with at least a `filename`
(relative to the image root) and a `caption`.

We provide **CC3M** as a worked example:

```
data_preparation/
└── cc3m/
    ├── download_cc3m.sh    # download images with img2dataset
    ├── build_data_info.py  # parquet metadata -> data_info.json
    └── README.md           # step-by-step guide
```

## Pipeline

```
Conceptual Captions        img2dataset            build_data_info.py
   (caption, url)   ─────►  cc3m_images/    ─────►   data_info.json
   Train_GCC*.tsv          {shard}/*.jpg            (filename, caption, ...)
                           {shard}.parquet
```

The produced `data_info.json` (plus the image root directory) is exactly the
input expected by [`../data_validation/compute_baseline_score`](../data_validation/compute_baseline_score).

## Requirements

```bash
pip install img2dataset pandas pyarrow
```

See [`cc3m/README.md`](cc3m/README.md) for the detailed CC3M instructions. Other
datasets (CC12M, LAION, YFCC15M, ...) follow the same recipe — produce a
`data_info.json` whose dicts contain `filename` and `caption`.
