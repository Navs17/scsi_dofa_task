# DOFA + Faster R-CNN on DIOR: Baseline vs. LoRA Fine-Tuning

This repository contains the implementation and experiments for the SCSI Lab task on downstream fine-tuning of the **DOFA remote-sensing foundation model** for object detection.

The project compares two models with an identical detection architecture:

1. **Baseline:** full fine-tuning of the DOFA backbone
2. **Modified:** LoRA-based fine-tuning of the DOFA backbone

The dataset, detection architecture, training conditions, and evaluation protocol are kept consistent between the two experiments so that the effect of the fine-tuning strategy can be compared directly.

---

## 1. Task Objective

The objective is to adapt a pretrained **DOFA (Dynamic One-For-All)** foundation model to remote-sensing object detection.

The experiment consists of:

```text
DIOR
  ↓
DOFA ViT-Base
  ↓
Simple Feature Pyramid
  ↓
Faster R-CNN
  ↓
Object detection
```

Two fine-tuning strategies are then compared:

```text
Baseline:
DOFA + Simple Feature Pyramid + Faster R-CNN
         ↓
    Full fine-tuning


Modified:
DOFA + Simple Feature Pyramid + Faster R-CNN
         ↓
      LoRA fine-tuning
```

The DOFA backbone remains the same pretrained foundation model in both experiments; only the fine-tuning strategy is changed.

---

## 2. References

### DOFA Paper

Xiong et al., **"Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation"**

https://arxiv.org/abs/2403.15356

### Official DOFA Repository

https://github.com/zhu-xlab/DOFA

### DOFA Hugging Face Model

https://huggingface.co/earthflow/DOFA

---

# 3. Dataset

## DIOR

The experiments use the **DIOR (Dataset for Object Detection in Optical Remote sensing images)** dataset.

Dataset source:

https://huggingface.co/datasets/HichTala/dior

DIOR contains:

* **23,463 images**
* **20 object categories**
* Horizontal bounding-box annotations
* Standard train/test/validation splits
* COCO-format annotations in the selected Hugging Face distribution

Example object categories include:

* Airplane
* Airport
* Bridge
* Chimney
* Dam
* Expressway service area
* Expressway toll station
* Golffield
* Ground track field
* Harbor
* Overpass
* Ship
* Stadium
* Storage tank
* Tennis court
* Train station
* Vehicle
* Windmill
* Baseball field
* Basketball court

### Why DIOR?

DIOR was selected because it is a widely used remote-sensing object-detection benchmark containing diverse object categories and horizontal bounding-box annotations. Its standard splits and COCO-compatible format also make it convenient for reproducible downstream detection experiments.

---

## 4. Training and Evaluation Subsets

Due to the available computational budget and the use of a single Tesla T4 GPU, the experiments were conducted using subsets of the full dataset.

### Training

A random subset of:

**4,000 images**

was sampled from the original **18,000-image training split** using:

```text
random seed = 42
```

Images were resized to:

```text
512 × 512
```

Training configuration:

```text
Epochs:       8
Batch size:   2
Image size:   512 × 512
```

### Evaluation

Evaluation was performed on:

**500 images from the standard DIOR test split**

The evaluation images were not taken from the training subset.

The same evaluation subset and evaluation procedure were used for both models.

> **Important:** The reported mAP values therefore represent performance on the 500-image evaluation subset, not the complete DIOR test set.

---

# 5. Model Architecture

The detection pipeline consists of three major components:

```text
Input RGB Image
     │
     ▼
┌───────────────────────┐
│ DOFA ViT-Base         │
│ Pretrained Backbone   │
└───────────────────────┘
     │
     ▼
Single-scale ViT feature map
     │
     ▼
┌──────────────────────────────┐
│ Simple Feature Pyramid       │
│ ViTDet-style feature pyramid │
└──────────────────────────────┘
     │
     ├── stride 4
     ├── stride 8
     ├── stride 16
     └── stride 32
     │
     ▼
┌───────────────────────┐
│ Faster R-CNN          │
│ RPN + RoI Heads       │
└───────────────────────┘
     │
     ▼
Class + Bounding Box Predictions
```

The detector predicts:

* 20 DIOR object classes
* Background

Therefore, the classification head operates over **21 classes including background**.

---

## 6. DOFA Backbone

The backbone is the pretrained **DOFA ViT-Base** model.

DOFA is a remote-sensing foundation model that uses wavelength-conditioned dynamic patch embedding to support different Earth-observation spectral configurations.

For standard RGB images, the wavelength values used in this implementation are:

```text
Red:   0.665 μm
Green: 0.560 μm
Blue:  0.490 μm
```

These values follow the RGB wavelength convention used in DOFA's published NAIP/aerial imagery examples.

### Spatial Feature Extraction

A standard Vision Transformer does not naturally provide the multi-scale feature hierarchy required by Faster R-CNN.

Instead of using the pooled classification representation, this implementation extracts the **spatial patch-level features** from DOFA.

At 512 × 512 input resolution, these features are transformed into a multi-scale pyramid.

---

# 7. Simple Feature Pyramid

The DOFA ViT backbone produces a spatial feature representation at a single effective scale.

A **ViTDet-style Simple Feature Pyramid** is therefore used as the detection neck.

The pyramid generates four feature levels:

```text
Level       Spatial resolution       Stride
------------------------------------------------
P2          128 × 128                4
P3           64 × 64                 8
P4           32 × 32                16
P5           16 × 16                 32
```

This provides multi-scale representations for detecting objects of different sizes.

The design follows the general principle of deriving a feature pyramid from a plain Vision Transformer backbone, as explored in:

> Li et al., *Exploring Plain Vision Transformer Backbones for Object Detection*

---

# 8. Detection Head

The detection head is implemented using **torchvision's Faster R-CNN**.

It consists of:

```text
Region Proposal Network (RPN)
          +
RoI feature extraction
          +
Classification head
          +
Bounding-box regression head
```

The implementation uses a custom DOFA-based backbone and a compatible multi-scale feature pyramid.

---

# 9. Why torchvision instead of MMDetection?

MMDetection was not used because of dependency compatibility concerns.

The available EC2 environment uses a recent PyTorch/CUDA configuration, while some MMDetection/MMCV components depend on precompiled native extensions tied to particular PyTorch and CUDA versions.

Using torchvision avoids this additional binary dependency and provides a native `FasterRCNN` implementation that supports custom backbones.

This keeps the implementation lightweight and reproducible within the available environment.

---

# 10. Baseline vs. LoRA

The two models use exactly the same:

* Dataset
* DOFA backbone architecture
* DOFA pretrained checkpoint
* Simple Feature Pyramid
* Faster R-CNN detector
* Input resolution
* Batch size
* Optimizer
* Learning rate
* Number of epochs
* Evaluation procedure

The **only intended experimental variable is the fine-tuning strategy applied to DOFA**.

| Component                    | Baseline               | Modified                 |
| ---------------------------- | ---------------------- | ------------------------ |
| Dataset                      | DIOR                   | DIOR                     |
| Backbone                     | DOFA ViT-Base          | DOFA ViT-Base            |
| Neck                         | Simple Feature Pyramid | Simple Feature Pyramid   |
| Detector                     | Faster R-CNN           | Faster R-CNN             |
| Fine-tuning                  | **Full fine-tuning**   | **LoRA**                 |
| LoRA rank                    | —                       | 8                        |
| LoRA alpha                   | —                       | 16                       |
| LoRA target                  | —                       | Attention `qkv` / `proj` |
| Backbone parameters          | Trainable               | Frozen except LoRA       |
| Approx. trainable parameters | 131.3M / 133.2M         | 20.6M / 132.5M           |

The LoRA experiment introduces low-rank trainable adapters into the attention layers while keeping the original DOFA backbone weights frozen.

This reduces the number of trainable backbone parameters while retaining the same overall detection architecture.

---

# 11. Experimental Environment

### Hardware

```text
Platform:       AWS EC2
GPU:            NVIDIA Tesla T4
GPU Memory:     15 GB
```

### Software

```text
Python:         3.13
PyTorch:        2.13
CUDA:           13.0
torchvision
timm:           0.9.2
```

The complete Python dependency list is provided in:

```text
requirements.txt
```

---

# 12. Training Configuration

Both experiments use the same basic training configuration:

```text
Optimizer:          AdamW
Learning rate:      1e-4
Weight decay:       1e-4
Batch size:         2
Epochs:             8
Image size:         512 × 512
LR scheduler:       None
Data augmentation:  None
Random seed:        42
```

The configuration files are:

```text
configs/
├── baseline.yaml
└── lora.yaml
```

---

# 13. Results

## 13.1 Quantitative Results

Evaluation was performed on a 500-image subset of the standard DIOR test split.

| Metric          | Baseline — Full FT | Modified — LoRA |
| --------------- | -----------------: | --------------: |
| mAP@[0.50:0.95] |             0.0405 |      **0.0436** |
| mAP@0.50        |             0.0809 |      **0.0875** |
| mAP@0.75        |             0.0357 |      **0.0377** |

Under the constrained training setup, LoRA produced slightly higher mAP at all three reported IoU evaluation settings.

The difference is relatively small, so the result should be interpreted as **comparable performance with substantially fewer trainable backbone parameters**, rather than as evidence that LoRA is universally superior to full fine-tuning.

---

# 14. Training Cost

### Per-epoch time and loss

| Epoch | Baseline loss | Baseline time | LoRA loss | LoRA time |
| ----: | ------------: | -------------: | --------: | ---------: |
| 0 | 0.8492 | 1173.9s | 0.8046 | 966.6s |
| 1 | 0.7700 | 1175.9s | 0.6552 | 966.8s |
| 2 | 0.6647 | 1175.4s | 0.5937 | 967.7s |
| 3 | 0.5879 | 1176.1s | 0.5476 | 968.4s |
| 4 | 0.5290 | 1175.0s | 0.5039 | 965.8s |
| 5 | 0.4829 | 1174.0s | 0.4655 | 965.4s |
| 6 | 0.4411 | 1172.8s | 0.4283 | 1597.9s |
| 7 | 0.4181 | 1171.7s | 0.3987 | 1433.5s |

| Metric | Baseline | LoRA |
|---|---|---|
| Time/epoch (average, all 8 epochs) | ~19.6 min | ~18.4 min |
| Time/epoch (epochs 0–5 only) | ~19.6 min | ~16.1 min |
| Final training loss | 0.4181 | 0.3987 |

Baseline epoch time was remarkably consistent across all 8 epochs (~19.6 min throughout). LoRA was noticeably **faster** for the first 6 epochs (~16.1 min vs. baseline's ~19.6 min) — plausibly because freezing most of the backbone reduces backward-pass work, even though the forward pass computes the same features through the full network. Epochs 6–7 saw an unexplained slowdown for LoRA (~26.6 min and ~23.9 min), which appears to be an environmental anomaly (e.g., background process or I/O contention on the instance) rather than an inherent property of LoRA, given how consistent the preceding six epochs were. Averaged across all 8 epochs, LoRA's mean epoch time (~18.4 min) was still slightly lower than the baseline's (~19.6 min).

LoRA also converged to a **lower training loss at nearly every epoch**, not just at the end — see the loss-curve chart in the presentation for the full trajectory.

---

# 15. Qualitative Results

Prediction visualizations are stored in:

```text
results/baseline/
results/lora/
```

The visualizations compare model predictions with the corresponding ground-truth bounding boxes.

Observed behavior includes:

### Common behavior

Both models successfully detect many instances of the dominant **storage tank** category, often with high confidence.

Both models also show difficulty detecting some rare categories, including **ships**.

### LoRA-specific observation

On some sparse scenes, such as images containing a single bridge, the LoRA model produced redundant overlapping proposals for the same object.

This behavior was not observed in the corresponding baseline prediction in the tested examples.

This may indicate a difference in RPN/detection behavior resulting from the altered gradient updates during LoRA fine-tuning. However, this observation requires further experiments before a causal conclusion can be made.

---

# 16. Interpretation

The main finding is:

> **LoRA achieved comparable, and slightly higher, detection performance than full fine-tuning under the constrained training setup while updating substantially fewer backbone parameters, and also trained faster on average and converged to a lower loss at nearly every epoch.**

Specifically:

* mAP@[0.50:0.95] increased from **0.0405 → 0.0436**
* mAP@0.50 increased from **0.0809 → 0.0875**
* mAP@0.75 increased from **0.0357 → 0.0377**
* Average epoch time decreased from **~19.6 min → ~18.4 min** (and was ~16.1 min for 6 of the 8 epochs)

At the same time, the LoRA configuration updates a much smaller fraction of the overall model parameters.

This suggests that parameter-efficient adaptation can be a viable strategy for adapting a large remote-sensing foundation model when training data and computational resources are limited — offering comparable or better accuracy, faster iteration in most epochs, and a much smaller set of trainable parameters.

However, the performance differences are small, two of the eight LoRA epochs showed an unexplained timing anomaly, and these results should not be interpreted as a definitive, general superiority of LoRA over full fine-tuning.

---

# 17. Limitations

Several limitations affect the absolute performance and generalizability of the results.

### Limited training data

Only 4,000 of the 18,000 available training images were used because of computational constraints.

### Limited training duration

The models were trained for only 8 epochs.

### No learning-rate schedule

A fixed learning rate was used rather than a scheduled learning-rate policy.

### No data augmentation

The experiments did not use augmentation such as:

* Horizontal flipping
* Random cropping
* Multi-scale training

### Limited evaluation

Evaluation was performed on a 500-image subset of the DIOR test split rather than the complete test set.

### Class imbalance

Randomly sampling 4,000 images may result in uneven representation of the 20 object categories. This may contribute to poor performance on less frequently represented classes.

### Low absolute mAP

The reported mAP values are substantially lower than results commonly reported for fully trained DIOR detection systems.

This is expected given the intentionally constrained training setup and should not be interpreted as a direct comparison with published DIOR benchmark results.

The primary objective of this experiment is the **controlled comparison between full fine-tuning and LoRA**, rather than achieving a state-of-the-art DIOR score.

### Unexplained training-time anomaly

LoRA epochs 6 and 7 took notably longer than epochs 0–5, for reasons not fully diagnosed (possible causes include background process contention or disk I/O interference on the shared EC2 instance). This should be re-verified with a clean, isolated re-run before drawing firm conclusions about LoRA's typical per-epoch speed.

---

# 18. Future Work

With additional computational resources and training time, the following experiments would be useful:

1. Train using the complete 18,000-image training set.
2. Increase the number of training epochs.
3. Introduce a learning-rate schedule.
4. Apply remote-sensing-specific data augmentation.
5. Investigate class-balanced or stratified sampling.
6. Evaluate on the complete DIOR test set.
7. Compare different LoRA ranks and target layers.
8. Investigate the redundant-box behavior observed in some LoRA predictions.
9. Compare LoRA with other parameter-efficient fine-tuning strategies.
10. Perform multiple random seeds to determine whether the small mAP difference is statistically robust.
11. Re-run LoRA training in isolation to confirm whether the epoch 6–7 slowdown recurs or was a one-off environmental artifact.

---

# 19. Reproducing the Experiments

## Install dependencies

```bash
pip install -r requirements.txt
```

## Baseline — Full Fine-Tuning

Train:

```bash
python3 train.py --config configs/baseline.yaml
```

Evaluate:

```bash
python3 evaluate.py \
    --config configs/baseline.yaml \
    --checkpoint experiments/baseline/epoch_7.pth \
    --max_images 500
```

Generate qualitative predictions:

```bash
python3 inference.py \
    --config configs/baseline.yaml \
    --checkpoint experiments/baseline/epoch_7.pth \
    --out_dir results/baseline \
    --num_images 6
```

---

## Modified Model — LoRA

Train:

```bash
python3 train.py --config configs/lora.yaml
```

Evaluate:

```bash
python3 evaluate.py \
    --config configs/lora.yaml \
    --checkpoint experiments/lora/epoch_7.pth \
    --max_images 500
```

Generate qualitative predictions:

```bash
python3 inference.py \
    --config configs/lora.yaml \
    --checkpoint experiments/lora/epoch_7.pth \
    --out_dir results/lora \
    --num_images 6
```

---

# 20. Repository Structure

```text
scsi-dofa-task/
│
├── README.md
├── requirements.txt
│
├── configs/
│   ├── baseline.yaml
│   └── lora.yaml
│
├── dataset/
│   ├── __init__.py
│   └── dior.py
│
├── models/
│   ├── __init__.py
│   ├── dofa_backbone.py
│   ├── detector.py
│   └── lora.py
│
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   ├── seed.py
│   └── visualization.py
│
├── train.py
├── evaluate.py
├── inference.py
│
├── data/
│   └── DIOR/
│
├── experiments/
│   ├── baseline/
│   └── lora/
│
├── results/
│   ├── baseline/
│   └── lora/
│
└── presentation/
```

Training checkpoints and generated results are excluded from version control where appropriate.

---

# 21. Reproducibility

The experiments use:

```text
Random seed: 42
```

and identical dataset/training/evaluation conditions for the baseline and LoRA experiments.

The configuration files provide the main experiment settings, while the training, evaluation, and inference scripts provide the executable pipeline.

---

# 22. Summary

This project demonstrates downstream adaptation of a pretrained **DOFA ViT-Base** foundation model for remote-sensing object detection on DIOR.

The baseline uses:

```text
DOFA + Simple Feature Pyramid + Faster R-CNN
+ Full Fine-Tuning
```

The modified model uses:

```text
DOFA + Simple Feature Pyramid + Faster R-CNN
+ LoRA Fine-Tuning
```

Under the constrained experimental setup, LoRA achieved slightly higher detection metrics and slightly faster average training time while updating substantially fewer backbone parameters.

The results demonstrate the feasibility of parameter-efficient adaptation of DOFA for downstream remote-sensing object detection, while also highlighting the need for larger-scale training, an isolated re-run to confirm the training-speed observation, and more extensive evaluation before drawing broader conclusions.