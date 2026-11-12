# FlowBraTS

**Brain tumor segmentation with incomplete multimodal MRI.** This repository contains **FlowBraTS v2.1**, built on M2FTrans v1 with modality–region modeling, conditional flow matching, and multiscale gated residual refinement. It supports all **15 nonempty modality combinations**.

This code-only release includes source, configurations, tests, and visualization scripts. Datasets, checkpoints, logs, predictions, and generated figures are excluded. The UniME-based v3.6 implementation is a separate project.

## Installation

Use Python 3.10 and CUDA-enabled PyTorch. Run commands from the repository root.

```bash
git clone git@github.com:orangeheyue/FlowBraTS.git
cd FlowBraTS
python -m venv .venv
source .venv/bin/activate
# Install a PyTorch build compatible with your CUDA environment first.
pip install -r requirements.txt
```

Required M2FTrans v1 source is bundled under `baselines/M2FTrans/M2FTrans_v1/`, with its original license. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Prepare data and weights

The default BraTS2020 layout is:

```text
preproced_dataset/
├── BraTS2020/
│   ├── vol/<case>_vol.npy
│   └── seg/<case>_seg.npy
└── Common_Dataset_Split/BraTS/BRATS2020_Training_none_npy/
    ├── train.txt
    ├── val.txt
    └── test.txt
pretrained_weights/M2FTrans/
└── m2ftransv1_brats2020_80.pth
```

| Item | Format |
|---|---|
| Images | float32, shape `(X, Y, Z, 4)` |
| Channel order | **FLAIR, T1ce, T1, T2** |
| Labels | uint8, shape `(X, Y, Z)`; map original label 4 to 3 |
| Split files | One case ID per line, without `_vol.npy`; disjoint train/val/test |
| BraTS2018 split 1 | `train1.txt`, `val1.txt`, `test1.txt` |
| Backbone weights | Dataset-matched M2FTrans **v1**, for 80³ input patches |

Follow M2FTrans's crop and normalization protocol. Reference scripts are in `baselines/M2FTrans/M2FTrans_v1/preprocess*.py`; edit their source/destination paths before running. They preserve historical preprocessing behavior, including its crop-boundary limitations.

For external data locations, edit `data.root`, `data.split_dir`, and `m2ftrans.checkpoint` in your YAML. Finetuning also uses `comparison.checkpoint` for the original baseline. Data and weights must be obtained separately.

## Train

### Pretrained warmup and finetuning

Configure the matching `finetune_brats<year>.yaml` and `finetune_brats<year>_warmup.yaml`, including batch size for your GPU.

```bash
python -m scripts.finetune_brats2020 --gpu 0
# For BraTS2018 or BraTS2021:
python -m scripts.finetune_dataset --year 2021 --gpu 0
```

The workflow uses **5 frozen-backbone warmup epochs**, then low-LR finetuning. The second phase resumes the warmup epoch counter; the default final epoch limit is 40, subject to early stopping. Logs and weights go to `outputs/finetune_brats<year>/{warmup,finetune}/`. Supervisors resume existing checkpoints.

### Single-stage training and continuation

```bash
CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config configs/v21_brats2020.yaml \
  --batch-size 1 --grad-accumulation 4 \
  --output outputs/v21_brats2020

CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config configs/v21_brats2020.yaml \
  --resume outputs/v21_brats2020/last.pt \
  --batch-size 1 --grad-accumulation 4 \
  --output outputs/v21_brats2020
```

Effective batch size = `batch_size × grad_accumulation`. Use a new output directory for an independent run. `--resume` restores training state; `--initialize` explicitly transfers older weights with a fresh optimizer and epoch counter. They are mutually exclusive.

| Configuration | Purpose |
|---|---|
| `finetune_brats{2018,2020,2021}*.yaml` | Pretrained warmup/finetuning |
| `v21_brats{2018,2020,2021}.yaml` | 300-epoch development runs; 2018/2021 default to random backbone initialization |
| `ablation_*.yaml` | Module ablations |
| `velocity_*.yaml` | Velocity-network comparisons |
| `et_sweep_*.yaml` | ET sampling, loss, and head experiments |
| `v21_brats2023_unime.yaml` | Uses the UniME data protocol; the backbone remains M2FTrans |

Original experimental settings are retained. Some presets use large batches or reference earlier checkpoints/cache reports; review these before running.

## Evaluate

```bash
# Validation
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --config configs/finetune_brats2020.yaml \
  --checkpoint outputs/finetune_brats2020/finetune/best.pt \
  --split val --output outputs/eval_val.csv

# Test after validation-based model selection
CUDA_VISIBLE_DEVICES=0 python -m scripts.evaluate \
  --config configs/finetune_brats2020.yaml \
  --checkpoint outputs/finetune_brats2020/finetune/best.pt \
  --split test --confirm-test --output outputs/eval_test.csv
```

Standalone evaluation covers all 15 combinations, reporting **WT/TC/ET/ETPro Dice and HD95** and their averages. Per-case results are saved in the corresponding `_per_mask/` directory. It reuses M2FTrans's 80³ sliding windows, 50% overlap, and metric conventions, and requires CUDA.

Modes: `infer` (default), `ungated`, and `baseline`. With a finetuned checkpoint, `baseline` evaluates that finetuned backbone. For the original pretrained M2FTrans, configure its matching `m2ftrans.checkpoint`, use `--mode baseline`, and omit `--checkpoint`.

Finetuning presets select checkpoints using **full-modality validation** and test all 15 combinations. Development presets generally validate all 15; the 2023 preset overrides this. Check each configuration's `validation` section when comparing experiments.

## Method and structure

```text
Available MRI + modality mask
  → M2FTrans encoding/fusion
  → modality–region graph + utility/uncertainty prediction
  → shared conditional Flow (default: Euler, NFE=2)
  → local residuals and spatial gates at 5³ / 10³ / 20³
  → M2FTrans decoder → segmentation
```

Training targets combine full-modality teacher features with label-driven task residuals. The teacher is a no-gradient pass of the current backbone, not a separately frozen or EMA teacher. Full-modality inputs also receive task-residual supervision. Inference uses only available images and their mask.

| Directory | Contents |
|---|---|
| `src/` | Models, data interfaces, losses, checkpoints, evaluation |
| `scripts/` | Training, evaluation, checks, ablations, launchers |
| `configs/` | YAML presets with `_base_` inheritance |
| `tests/` | Unit tests and protocol checks |
| `Visual/` | Visualization and CAM scripts |
| `baselines/` | Required M2FTrans v1 source and license |

Visualization dependencies: `pip install -r requirements-visual.txt`. Multi-baseline figures additionally require RFNet/DiffUNet source and weights; configure their paths in `Visual/common.py`.

## Checks

```bash
# No datasets or pretrained weights required
OMP_NUM_THREADS=4 python -m pytest -q

# Two-batch check after preparing real data and backbone weights
CUDA_VISIBLE_DEVICES=0 python -m scripts.train \
  --config configs/v21_brats2020.yaml --epochs 1 \
  --batch-size 1 --grad-accumulation 1 --smoke-batches 2 \
  --output outputs/smoke
```

Smoke checkpoints are marked `smoke_only` and rejected by formal evaluation. Passing implementation checks is not an accuracy claim.

## License

FlowBraTS uses the [MIT License](LICENSE). Bundled M2FTrans code retains its [original MIT License](baselines/M2FTrans/LICENSE). Datasets and checkpoints are subject to their providers' terms.
