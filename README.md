# Object Detection and Instance Segmentation on COCO

**Case Study**
**Author:** Theophilus Frimpong

---

## Overview

This repository contains the full implementation and experimental results for a comparative study of **Faster R-CNN** (object detection) and **Mask R-CNN** (instance segmentation) on a subset of the COCO 2017 benchmark. Both models use a ResNet-50 + FPN backbone and were trained for 10 epochs on an NVIDIA L40S GPU via the UTRGV Cradle HPC cluster. To save space, only epoch 10 .pth was saved on the zip file, but the full code saves each epoch.

## Project Structure

```
COCO_CASE_STUDY/
├── data/
│   ├── build_subset.ipynb          # Notebook to download & build COCO subset
│   ├── coco/                       # Raw extracted COCO 2017 archives (not tracked)
│   ├── coco_downloads/             # Downloaded zip files (not tracked)
│   └── coco_subset/                # Final subset used for all experiments
│       ├── train2017/              # 3,000 sampled training images
│       ├── val2017/                # 500 sampled validation images
│       └── annotations/
│           ├── instances_train2017_subset.json
│           └── instances_val2017_subset.json
│
├── faster_rcnn/
│   ├── scripts/
│   │   ├── dataset.py              # COCO dataset loader
│   │   ├── model.py                # Faster R-CNN model setup
│   │   ├── utils.py                # Helper functions
│   │   ├── train.py                # Training loop
│   │   ├── eval.py                 # COCO evaluation (pycocotools)
│   │   └── visualize.py            # Prediction visualization
│   ├── slurm/
│   │   └── run_frcnn.slurm         # Slurm job script for cluster
│   ├── logs/                       # Training logs
│   ├── results/                    # JSON result files
│   └── figures/                    # Output figures
│
├── mask_rcnn/
│   ├── scripts/
│   │   ├── dataset.py
│   │   ├── model.py
│   │   ├── utils.py
│   │   ├── train.py
│   │   ├── eval.py
│   │   └── visualize.py
│   ├── slurm/
│   │   └── run_maskrcnn.slurm      # Slurm job script for cluster
│   ├── logs/
│   ├── results/
│   └── figures/
│
└── Guide.txt                       # Step-by-step guide for running on a cluster
```

---

## Dataset

### How the subset was created

The full COCO 2017 dataset was downloaded and extracted **locally** using `data/build_subset.ipynb`. The notebook handles three steps:

1. **Download** — fetches `train2017.zip`, `val2017.zip`, and `annotations_trainval2017.zip` from the official COCO servers using `urllib`.
2. **Extract** — unzips all archives into `data/coco/`.
3. **Subset creation** — with `random.seed(42)`, randomly samples **3,000 image IDs** from the training split and **500 image IDs** from the validation split. Only the selected images are copied, and new filtered annotation JSON files are written containing only those images and their corresponding annotations. The full 80-category list is preserved.

The raw downloads and fully extracted archives are not tracked in this repository (they total ~20 GB). Raw data can be directly downloaded using the build_subset.ipynb file.

### Reproducibility

The exact subset used for all experiments is available directly in `data/coco_subset/`. Both the images and the filtered annotation JSON files are included. No re-downloading or re-sampling is needed to reproduce results — simply point the training and evaluation scripts at `data/coco_subset/` and the results will be identical (same images, same annotations, same random seed).

---

## Getting Started

### Requirements

```bash
pip install torch torchvision torchaudio
pip install pycocotools matplotlib opencv-python pillow tqdm
```

Tested with PyTorch 2.x, CUDA 12.1, Python 3.10.

### Running on a cluster (UTRGV Cradle / TACC)

See **`Guide.txt`** for a full walkthrough covering environment setup, file transfer, job submission, and monitoring. Slurm scripts are provided in each model's `slurm/` directory and can be submitted directly:

---

## Training Configuration

| Setting | Value |
|---|---|
| GPU | NVIDIA L40S |
| Backbone | ResNet-50 + FPN |
| Pretrained weights | COCO (torchvision defaults) |
| Optimizer | SGD, momentum=0.9, weight decay=5e-4 |
| Learning rate | 0.005, StepLR decay at epoch 7 (γ=0.1) |
| Batch size | 2 |
| Epochs | 10 |
| Mixed precision | Enabled (AMP) |
| Augmentation | Random horizontal flip (p=0.5) |
| Random seed | 42 |

---

## Results

Full per-epoch metrics, RPN recall tables, training/validation loss curves, and qualitative detection figures are available in each model's `results/` and `figures/` directories.

---

## License

This project was developed for academic purposes as part of MATH 8335 at UTRGV.
