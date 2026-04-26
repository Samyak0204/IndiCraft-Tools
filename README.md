#  Deep Learning for Computer Vision — Phase 2
### *Understanding Inductive Bias, Attention Mechanisms, and Fine-Tuning in Vision Models*

**Dataset:** IndiCraft Tools (8-class)  |  **Author:** Samyak Chhabra  |  **Date:** April 2026

---

## Overview

This repository contains three mechanistic experiments on modern vision models, all conducted on the **IndiCraft Tools Dataset** — an 8-class benchmark of handcrafted Indian tools. The emphasis is on *causal explanation* of model behavior rather than raw performance benchmarking.

Each experiment isolates a single variable under a strict one-factor-at-a-time (OFAT) protocol, ensuring observed differences are attributable to the manipulated variable alone.

---

## Repository Structure

```
dlcv_phase_2/
├── code/
│   ├── unit1.ipynb          # Problem 1: Data Augmentation as Inductive Bias
│   ├── unit2.ipynb          # Problem 2: Attention Surgery Experiment (ViT)
│   └── unit3.ipynb          # Problem 3: Transfer Learning Freeze Strategies
│
├── results/
│   ├── unit1/
│   │   ├── heatmap_overall.png          # Augmentation × Corruption accuracy matrix
│   │   ├── heatmap_perclass_clean.png   # Per-class accuracy (failure case analysis)
│   │   ├── robustness_gap.png           # Clean → corrupted accuracy drop
│   │   ├── ablation_clean_acc.png       # Atomic vs combined augmentation comparison
│   │   ├── training_curves.png          # Loss + val accuracy across all strategies
│   │   ├── failure_analysis.txt         # Structured failure report
│   │   └── results.json
│   ├── unit2/
│   │   ├── head_importance_map.png      # 12×6 head importance heatmap (main deliverable)
│   │   ├── results.json
│   │   └── {Task}_failure_{n}.png       # Failure cases per task (Texture/Shape/Spatial)
│   └── unit3/
│       ├── freezing_strategy_curves.png # Train/val curves per freeze strategy
│       ├── overfitting_gap.png          # Overfit gap across epochs
│       ├── training_summary.png         # Final-epoch metric comparison
│       └── results.json
│
└──IndiCraft-crops-20260426T130049Z-3-001 # Cropped Images
```

---

## Experiments

### Unit 1 — Data Augmentation as Inductive Bias

> **Question:** Which augmentations actually help, and why?

**Model:** ResNet-18 (identical config across all runs)

**Hypothesis:** Augmentation does not merely increase data volume — it encodes explicit invariances into the learned representation. A model becomes robust only to the transformations it was trained to ignore.

**Augmentation strategies tested (atomic + combined ablation):**

| Strategy | Type |
|---|---|
| Baseline (no aug) | Reference |
| Rotation only | Atomic |
| Crop only | Atomic |
| Flip only | Atomic |
| Color Jitter only | Atomic |
| Cutout only | Atomic |
| Geometric Combo | Combined |
| Color Combo | Combined |
| Full Mix | Combined |

**Evaluation:** Clean test set + 4 corruptions (Gaussian noise, blur, brightness shift, low contrast)

**Key findings:**

- Clean accuracy is nearly identical (~47%) across all strategies — augmentation changes *what* is learned, not *how much*
- Geometric augmentation improves noise robustness (Rotation: 43.2% vs Baseline: 40.5%) by enforcing structural feature reliance
- **Blur is the universal failure mode** — no strategy trains for blur, so none achieves blur invariance
- Full Mix performs *worst* on blur (14.1%) due to conflicting inductive biases causing gradient interference
- Class imbalance dominates per-class metrics: `carpentry_hammer` and `hand_planer` collapse to near-zero across all strategies

**Selected results (Augmentation × Corruption matrix):**

| Strategy | Clean% | Noise% | Blur% | Bright% | Contrast% | Avg Gap▲ |
|---|---|---|---|---|---|---|
| Baseline | 47.6 | 40.5 | 25.6 | 46.2 | 44.4 | 8.4 |
| Rotation only | 47.1 | 43.2 | 17.8 | 45.9 | 41.9 | 9.9 |
| Color Jitter only | 47.6 | 41.7 | 20.1 | 47.1 | 44.6 | 9.2 |
| Color Combo | 47.7 | 39.0 | 22.6 | 47.7 | 46.7 | 8.7 |
| **Full Mix** | 46.6 | 35.8 | **14.1** | 45.4 | 46.2 | **11.2** |

---

### Unit 2 — The Attention Surgery Experiment

> **Question:** What happens when you remove specific attention heads in ViTs?

**Model:** ViT-B/16 pretrained on ImageNet (12 layers × 6 heads = 72 ablation experiments)

**Method:** Each head is replaced with a uniform attention distribution (zero selective attention). A linear probe measures the accuracy drop — isolating the causal contribution of each head without fine-tuning confounds.

**Three task-specific evaluation axes:**

| Task | Transform Applied | Probes For |
|---|---|---|
| Texture removed | Gaussian blur (kernel=9) | Which heads rely on texture cues |
| Shape only | Grayscale (3-channel) | Which heads encode shape/silhouette |
| Spatial localization | Fixed pad+crop shift | Which heads encode positional layout |

**Key findings:**

- No single head causes catastrophic failure — max drop is 4.5%, confirming **distributed redundant representations**
- **Spatial heads are most critical** (L4-H2, L7-H1: 4.5% drop each) — ViTs are fundamentally patch-relational reasoners
- **Texture heads are most redundant** (max ~2% drop, concentrated in early layers L1–L3)
- **Shape heads are maximally distributed** — the primary classification signal is protected against single-head failure
- Several heads show *negative* drops (removal improves accuracy by 0.5–1.0%) — suppressive heads encoding spurious co-variates

**Head importance summary:**

| Task | Max Drop | Most Critical Heads |
|---|---|---|
| Spatial Localization | **4.5%** | L4-H2, L7-H1, L6-H1 |
| Texture Recognition | 2.0% | L1-H2, L3-H4 |
| Shape Recognition | 2.0% | L1-H1, L2-H1 |

---

### Unit 3 — Transfer Learning Layer Freezing Strategy

> **Question:** Which layers should you freeze when fine-tuning a pretrained model on small data?

**Model:** ImageNet-pretrained ResNet-50 fine-tuned on IndiCraft Tools

**Hypothesis:** Early layers learn universal, domain-agnostic features (edges, corners, textures). Late layers learn task-specific semantics that must be adapted to the target domain.

**Strategies compared:**

| Strategy | Frozen Layers | Test Acc% | Train Time (s) | Overfit Gap |
|---|---|---|---|---|
| **Freeze Early (L1+L2)** ★ | L1, L2 | **46.7** | 3,475 | 0.542 |
| Full Fine-Tune | None | 45.7 | 5,487 | 0.528 |
| Head Only | All backbone | 43.2 | 2,104 | 0.453 |
| Freeze Late (L3+L4) | L3, L4 | 41.7 | 5,073 | 0.478 |

**Key findings:**

- **Freeze Early wins**: +1% accuracy over full fine-tuning, **37% faster training** (3,475s vs 5,487s)
- Full fine-tuning overfits severely (train acc 97.3%, test acc 45.7%) — 25M parameters on a small dataset is a capacity–data mismatch
- Freeze Late is the worst strategy: prevents semantic adaptation while consuming nearly as much compute as full fine-tuning
- Head-only is best for extreme compute constraints: fastest (421s/epoch) with lowest overfit gap

**Recommendations by scenario:**

| Scenario | Strategy |
|---|---|
| Small dataset, similar domain | **Freeze Early (L1+L2)** |
| Very limited compute | Head Only |
| Large dataset available | Full Fine-Tune |
| Very different domain | Freeze Early + LR warmup |
| Any scenario |  Avoid Freeze Late |

---

## Cross-Experiment Synthesis

All three experiments are unified by a single principle: **the structure of learned representations is determined by the constraints imposed during training.**

1. **Augmentation** explicitly encodes which input transformations are class-irrelevant → models generalize only along those axes
2. **Self-attention** imposes a global relational inductive bias → ViTs are spatial reasoners, not local texture detectors
3. **Pretraining** encodes a powerful image-domain prior in early layers → freezing them transfers this prior efficiently

A consistent finding across Units 1 and 2: deep models achieve robustness through **redundancy**. No single augmentation dominates all corruption types; no single attention head is individually catastrophic. This redundancy is both a strength (fault tolerance) and a challenge (interpretability and compression).

---

## Setup and Requirements

```bash
# Clone the repository
git clone https://github.com/Samyak0204/IndiCraft-Tools.git
cd IndiCraft-Tools

# Install dependencies
pip install torch torchvision timm seaborn matplotlib numpy Pillow

# Run experiments (Google Colab recommended for GPU)
# Update CROPS_ROOT in each notebook to your dataset path
```

**Dataset path structure expected:**
```
IndiCraft-crops/
├── train/
│   └── <class_name>/   (8 classes)
├── val/
│   └── <class_name>/
└── test/
    └── <class_name>/
```

**Hardware:** All experiments run on a single GPU (CUDA). CPU fallback is supported but slow for Unit 2 (72 ablation passes).

---

## Key Takeaways

- **Augment for the corruption you care about** — mismatched augmentation provides no robustness benefit
- **ViTs are spatial reasoners first** — optimize head pruning/compression around spatial heads, not texture heads
- **Always freeze early layers** on small datasets — universal low-level features are not worth relearning
- **Address class imbalance before model-level interventions** — minority class failure persists regardless of architecture or augmentation strategy

---

## Citation / Reference

```
Chhabra, S. (2026). Deep Learning for Computer Vision — Phase 2:
Understanding Inductive Bias, Attention Mechanisms, and Fine-Tuning
in Vision Models. IndiCraft Tools Dataset.
```
