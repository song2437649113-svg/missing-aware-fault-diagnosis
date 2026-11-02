# Missing-aware multi-domain bearing fault diagnosis

Core code, trained weights and a runnable demo for missing-aware, multi-domain fault diagnosis of rolling bearings from tribovoltaic signals.


## Overview

Tribovoltaic signals recorded under different operating domains can contain missing segments. This repository provides the core implementation of a missing-aware diagnosis framework with four stages:

1. **Multi-domain signals.** Each 4,096-sample window is split into 16-sample patch tokens that keep their original positions.
2. **Index-aware MAE.** A masked autoencoder with rotary position encoding (RoPE) applied to the physical patch indices is pretrained by masked reconstruction and then frozen. It fills missing patches, while observed samples are kept unchanged.
3. **Dual-branch diagnosis.** Frozen MAE features and Welch log-power spectral densities are fused by five independent heads into 64-D unit embeddings, which are compared with cross-domain class prototypes.
4. **Reliable decisions.** Ensemble risk across the five heads routes each prediction to *accept* or *review*, and a novelty score computed from the native features flags out-of-distribution (OOD) candidates.

## Repository structure

```
├── demo.ipynb              # end-to-end demo: missing input → MAE completion → diagnosis → accept / review / OOD
├── src/
│   ├── config.py           # settings used in the paper
│   ├── mae.py              # index-aware MAE with RoPE: model, weight loading, pretraining
│   ├── completion.py       # MAE completion of missing patches
│   ├── features.py         # inputs of the two branches: MAE features and Welch log-PSD
│   └── diagnosis.py        # fusion heads, class prototypes and the decision layer
├── weights/
│   ├── rope_mae_pretrained.pth            # pretrained index-aware MAE
│   └── fault_diagnosis_uq_augmented.pth   # trained diagnosis model (includes the frozen MAE)
├── data/                   # demo set (see below)
├── requirements.txt
└── LICENSE
```

## System requirements

- **Software:** Python 3 with PyTorch, NumPy, SciPy, pandas, Matplotlib and Jupyter (see `requirements.txt`). The demo was tested with Python 3.13, PyTorch 2.14.1 (CPU), NumPy 2.5.3, SciPy 1.18.1, pandas 3.0.5 and Matplotlib 3.11.2; the models were trained with Python 3.12, PyTorch 2.14.1, NumPy 2.3.5, SciPy 1.18.1 and pandas 3.0.1.

## Installation guide

```bash
git clone https://github.com/song2437649113-svg/missing-aware-fault-diagnosis.git
cd missing-aware-fault-diagnosis
pip install -r requirements.txt
```

## Demo

### Data

`data/` contains 210 windows of the validation split used in the paper: 10 windows per class and operating domain for the six known classes (C0–C5) and 10 windows per domain of a fault type that was never used for training (U).

| File | Shape | Content |
|---|---|---|
| `x_demo.npy` | (210, 4096) | tribovoltaic signal windows, float32 |
| `y_demo.npy` | (210,) | class label: 0–5 known, 6 unknown |
| `d_demo.npy` | (210,) | operating domain: 0–2 |

### How to run

From the repository root:

```bash
jupyter notebook demo.ipynb
```

Run all cells in order. The notebook

1. removes 60% of the 16-sample patches of each window at random and fills them with the frozen MAE;
2. removes one contiguous segment covering 0, 5, 10 or 15% of each window (three random positions per window), fills it with the MAE, and passes the completed windows through the dual-branch classifier and the decision layer.

### Expected output

The notebook in this repository is saved with its outputs, which include:

- the masked RMSE of the MAE completion (0.0787 over the 180 known windows with 60.2% of the samples removed) and an example window;
- the following summary, in % of the 180 known or 30 unknown windows (mean over three masks):

| Missing input (%) | Known: accuracy | Known: accepted | Known: review | Known: OOD | Accepted accuracy | Unknown: accepted | Unknown: review | Unknown: OOD |
|---|---|---|---|---|---|---|---|---|
| 0 | 100.0 | 97.2 | 0.0 | 2.8 | 100.0 | 0.0 | 0.0 | 100.0 |
| 5.08 | 100.0 | 78.7 | 0.0 | 21.3 | 100.0 | 0.0 | 0.0 | 100.0 |
| 10.16 | 99.6 | 96.7 | 0.0 | 3.3 | 99.6 | 0.0 | 0.0 | 100.0 |
| 14.84 | 99.8 | 98.9 | 0.0 | 1.1 | 99.8 | 0.0 | 0.0 | 100.0 |

*Accuracy* is computed before rejection; *accepted accuracy* is the accuracy of the known windows that were accepted.

## Instructions for use

To run the trained model on other windows from the same test rig, store them as a float32 array of shape (n, 4096), mark missing samples as NaN in whole 16-sample patches, and provide the operating domain (0, 1 or 2) of each window:

```python
import numpy as np
from src.mae import load_configured_mae
from src.completion import mae_complete
from src.diagnosis import load_diagnosis_model

mae, _ = load_configured_mae("weights/rope_mae_pretrained.pth")
model = load_diagnosis_model("weights/fault_diagnosis_uq_augmented.pth")

patch_mask = np.isnan(x).reshape(len(x), -1, 16).all(-1)   # True = missing patch
x_filled = mae_complete(mae, x, patch_mask)                 # observed samples are unchanged
out = model.predict(x_filled, domains)
out["decision"]   # predicted class 0-5 if accepted, -2 = review, -1 = OOD candidate
```

`src/mae.py` also contains the self-supervised pretraining loop (`pretrain`). The full dataset and the training script of the diagnosis heads are available from the corresponding author upon reasonable request.

## License

This project is released under the MIT License.