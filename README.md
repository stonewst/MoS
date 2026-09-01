

# Mixture-of-Scores: Robust Image-Text Data Valuation via Three Lines of Code

**ICCV 2025**

Sitong Wu, Haoru Tan, Yukang Chen, Shaofeng Zhang, Jingyao Li, Bei Yu, Xiaojuan Qi, Jiaya Jia

[![Paper](https://img.shields.io/badge/ICCV-2025-blue)](https://openaccess.thecvf.com/content/ICCV2025/papers/Wu_Mixture-of-Scores_Robust_Image-Text_Data_Valuation_via_Three_Lines_of_Code_ICCV_2025_paper.pdf)
[![Code](https://img.shields.io/badge/GitHub-MoS-black?logo=github)](https://github.com/stonewst/MoS)



## Introduction

**Mixture-of-Scores (MoS)** is a simple yet effective method for **robust image-text data valuation** in vision-language pre-training.

Most image-text data-valuation metrics use an off-the-shelf model (e.g. CLIP-Score) to score a pair by feature similarity. However, **different scoring models produce inconsistent quality scores for the same data**, and no single score excels across all tasks. MoS integrates multiple existing quality scores into a robust ensemble score via a data-adaptive strategy, mitigating the bias of any single score, and it takes only **three lines of code**.

Concretely, for the $i$-th image-text pair with baseline scores $\{S^1_i, ..., S^M_i\}$, MoS computes a weighted sum where each weight reflects (1) the **consensus** of a score among all models and (2) the **quality uncertainty** of the data point:

```python
# s: [N, M] baseline scores of N pairs from M scoring models
density = -(s[:, :, None] - s[:, None, :]).abs().sum(-1) / (M - 1)     # consensus
temp    = (tmax - tmin) / (std.max() - std.min()) * std + (tmin * std.max() - tmax * std.min()) / (std.max() - std.min())  # uncertainty-aware temperature
mos     = (torch.softmax(density / temp[:, None], dim=-1) * s).sum(-1) # data-adaptive ensemble
```

These three lines are implemented in [`data_validation/compute_mos/mos_core.py`](data_validation/compute_mos/mos_core.py).

**Highlights**

- 🔍 **A new problem**: the first to discover and investigate the *quality score disparity* of image-text data and its impact on vision-language pre-training.
- 🧩 **Data-adaptive ensemble**: weights each baseline score by its *consensus* among models and the data's *quality uncertainty*, harnessing complementary strengths while mitigating individual bias.
- ⚡ **Three lines of code**: negligible cost (~2 min to ensemble 18 scores over 1M pairs on a single GPU).



## Repository structure

```
MoS/
├── data_preparation/             # download & convert a dataset (CC3M example) into data_info.json
├── data_validation/              # produce a robust quality score for image-text data
│   ├── compute_baseline_score/   #   Step 1: score with several VLMs (CLIP, BLIP, ...) + merge
│   └── compute_mos/              #   Step 2: ensemble the baseline scores -> MoS score
└── data_filtering/               # sort by score, keep the top k% as training data
```



## Installation

We use a single conda environment named `mos` (tested on 8×H200, CUDA 12.1):

```bash
conda create -n mos python=3.10 -y
conda activate mos

# PyTorch (CUDA 12.1)
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu121

# shared dependencies
pip install -r requirements.txt
```

The four scoring backends share one env except `timm`, which must be swapped
between the EVA-CLIP and BLIP2 scorers. See [installation.md](installation.md)
for the full setup and version notes.

## Data Preparation

We use **CC3M** as an example. Download the images with `img2dataset` and
convert the metadata into a `data_info.json` (a list of dicts, each with a
`filename` and a `caption`):

```bash
cd data_preparation/cc3m
bash download_cc3m.sh                                          # see its README
python build_data_info.py --img2dataset_dir /path/to/cc3m_images \
                          --output_path /path/to/cc3m/data_info.json
```

See [data_preparation/README.md](data_preparation/README.md) for details and
other datasets.

## Data Validation

**1) Compute baseline quality scores** with several VLMs, then merge them:

```bash
cd data_validation/compute_baseline_score
python compute_clip_score.py --clip_model_name ViT-B/32 ...    # see its README
python compute_blip_score.py --vit_arch large ...
python merge_scores.py join --inputs <...> --output data_info_all_scores.json
```

See [data_validation/compute_baseline_score/README.md](data_validation/compute_baseline_score/README.md) for details.


**2) Ensemble into the robust MoS score:**

```bash
cd ../compute_mos
python compute_mos.py \
    --data_info_path data_info_all_scores.json \
    --output_path    data_info_mos.json \
    --auto_keys --save_key mos_score
```

See [data_validation/compute_mos/README.md](data_validation/compute_mos/README.md) for details.

## Data Curation

The MoS score can drive various data-processing strategies before pre-training,
such as **data filtering**, **sample weighting**, and **image re-captioning**.
As an example, we provide **data filtering** here, which keeps the top k%
highest-scoring pairs as the training set:

```bash
cd data_filtering
python filter_data.py \
    --data_info_path /path/to/data_info_mos.json \
    --output_path    /path/to/data_info_filtered.json \
    --score_key mos_score --keep_ratio 0.8
```

See [data_filtering/README.md](data_filtering/README.md) for more options
(e.g. keeping the lowest-scoring subset or dumping the removed data).

## Training

We use the [DeCLIP](https://github.com/sense-gvt/declip) codebase for
vision-language pre-training. Feed the curated `data_info` (e.g.
`data_info_filtered.json`) into DeCLIP and follow its official instructions.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{wu2025mos,
  title={Mixture-of-Scores: Robust Image-Text Data Valuation via Three Lines of Code},
  author={Wu, Sitong and Tan, Haoru and Chen, Yukang and Zhang, Shaofeng and Li, Jingyao and Yu, Bei and Qi, Xiaojuan and Jia, Jiaya},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2025}
}
```



## Acknowledgement

This codebase builds on several excellent open-source projects:
[OpenAI CLIP](https://github.com/openai/CLIP),
[open_clip](https://github.com/mlfoundations/open_clip) (EVA-CLIP),
[BLIP](https://github.com/salesforce/BLIP),
[LAVIS](https://github.com/salesforce/LAVIS) (BLIP2),
[img2dataset](https://github.com/rom1504/img2dataset), and
[DeCLIP](https://github.com/Sense-GVT/DeCLIP). We thank the authors and
community for their contributions.


## Contact

For questions about the paper or code, please contact **Sitong Wu** at
[stonewst@163.com](mailto:stonewst@163.com), or open an issue on
[GitHub](https://github.com/stonewst/MoS).