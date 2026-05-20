# HAF: Hamiltonian Asymmetric Fusion

**Hamiltonian Asymmetric Fusion: One-Way Safe Directed Refinement under Modality Imbalance**

This repository contains the implementation of **Hamiltonian Asymmetric Fusion (HAF)**, a lightweight directed fusion module designed for multimodal salient object detection under **modality imbalance**.

---

## Abstract

In RGB-D salient object detection, multimodal fusion is commonly implemented through symmetric token interaction, which implicitly allows information to flow in both directions. Under **modality imbalance**, when the auxiliary stream is substantially noisier than the designated primary stream, such symmetric fusion may introduce a harmful **backflow channel** that injects auxiliary noise into the primary representation and amplifies errors during iterative refinement.

We propose **Hamiltonian Asymmetric Fusion (HAF)**, a lightweight unrolled refinement module that performs **one-way safe directed refinement**. The primary modality provides stable guidance, while the auxiliary modality is iteratively refined with momentum regularization and gated driving. The refinement force is instantiated by FFT-based spectral global correlation and modulated by a learnable spectral response to emphasize reliable frequency components. HAF is designed to achieve stable multimodal fusion while avoiding the error amplification caused by symmetric cross-modal interaction.

---

## Method Framework

The overall framework of HAF is shown below. HAF formulates multimodal fusion as an unrolled Hamiltonian-style refinement process, where the primary modality provides stable guidance and the auxiliary modality is refined through gated force-driven updates.

### Comparison between CDA and HAF

- [Figure 1: Comparison between CDA and HAF](assets/img_cda_haf.pdf)

### Overall Pipeline of HAF

- [Figure 2: Overall pipeline of HAF](assets/img_haf_overall.pdf)

---

## Dependencies

The code is implemented with PyTorch. The main dependencies include:

```text
python
torch
torchvision
timm
numpy
```

If CUDA-specific PyTorch is required, please install PyTorch according to your CUDA version from the official PyTorch website.

---


## Dataset Preparation

Please organize the RGB-D salient object detection datasets as follows:

```text
datasets/
├── DUT-RGBD/
├── LFSD/
├── NJU2K/
├── NLPR/
├── SIP/
└── STERE/

```

---

## Pretrained Model

We use **Swin-B** as the pretrained backbone.

Please download the pretrained Swin-B weights and place them in the following directory:

```text
pretrained/
└── swin_base_patch4_window12_384_22k.pth
```

After downloading the Swin-B pretrained model, the project directory should look like:

```text
HAF/
├── models/
├── datasets/
├── pretrained/
│   └── swin_base_patch4_window12_384_22k.pth
├── assets/
├── train.py
├── test.py
└── README.md
```