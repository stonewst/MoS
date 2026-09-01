# Installation

MoS uses a **single** conda environment named `mos` for the whole pipeline
(data preparation -> baseline scoring -> MoS). The four baseline scoring
backends share one env, with a single mutually incompatible pin (`timm`), so we
keep one env and swap only that package when switching between the **EVA-CLIP**
and **BLIP2** scorers. See [Version notes](#version-notes-important).

## 1. Create the environment

```bash
source /path/to/anaconda3/bin/activate
conda create -y -n mos python=3.10
conda activate mos
```

## 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu121
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

## 3. Install the shared dependencies

```bash
pip install -r requirements.txt
```

This installs the common packages (data preparation + the shared scoring deps,
including the OpenAI `clip` package). It intentionally does **not** install
`open_clip` (EVA-CLIP) or `salesforce-lavis` (BLIP2), because those two pin
conflicting `timm` versions and cannot be resolved together.

## 4. Install the swappable scorers

`open_clip` needs `timm>=1.0.17` while `salesforce-lavis` pins `timm==0.4.12`,
so only one of them can be active at a time. Install whichever you need:

```bash
# EVA-CLIP:
pip install "open_clip_torch==3.3.0" "timm>=1.0.17"

# BLIP2 (LAVIS):
pip install "salesforce-lavis==1.0.2" "timm==0.4.12"
```

`salesforce-lavis` also pulls `opencv-python-headless==4.5.5.64`, which requires
`numpy<2` (already pinned in `requirements.txt`).

BLIP uses the official repo (cloned, not pip-installed):

```bash
git clone https://github.com/salesforce/BLIP.git
```

## Version notes (important)

All four scorers share one env **except `timm`**:

| Backend                | Requirement                                    |
|------------------------|------------------------------------------------|
| CLIP (OpenAI)          | any `timm`                                      |
| EVA-CLIP (`open_clip`) | `timm>=1.0.17`                                  |
| BLIP (official repo)   | any `timm`                                       |
| BLIP2 (LAVIS)          | `timm==0.4.12`, `transformers<4.27`, `numpy<2`  |

`numpy<2` and `transformers<4.27` satisfy **all four** backends, so they are
pinned once (in `requirements.txt`) and never change. The only package to swap
is `timm`:

```bash
# before EVA-CLIP:
pip install "timm>=1.0.17" "numpy<2"

# before BLIP2:
pip install "timm==0.4.12" "numpy<2"
```

CLIP and BLIP run under either `timm`, so they need no swap.

## Tested versions (8xH200, CUDA 12.1)

`torch 2.4.0+cu121`, `torchvision 0.19.0+cu121`, `numpy 1.26.4`,
`transformers 4.26.1`, `open_clip_torch 3.3.0`, `salesforce-lavis 1.0.2`,
`opencv-python-headless 4.5.5.64`, `img2dataset 1.47.0`, `pandas 2.3.3`,
`pyarrow 25.0.1`. EVA-CLIP verified with `timm 1.0.29`; BLIP2 with `timm 0.4.12`.

## Note on user site-packages

Always run with the `mos` env active. Packages installed via `pip install --user`
live in `~/.local` and can **shadow** the conda env (e.g. a different `timm`
version). Set `export PYTHONNOUSERSITE=1` to make Python ignore `~/.local`.
