# MMDisCo (ICLR 2025)

[![arXiv](https://img.shields.io/badge/arXiv-2405.17842-b31b1b.svg)](https://arxiv.org/abs/2405.17842)

This repository is the official implementation of **["MMDisCo: Multi-Modal Discriminator-Guided Cooperative Diffusion for Joint Audio and Video Generation (ICLR 2025)"](https://arxiv.org/abs/2405.17842)**.

[Akio Hayakawa](https://scholar.google.com/citations?user=sXAjHFIAAAAJ)<sup>1</sup>, [Masato Ishii](https://scholar.google.co.jp/citations?user=RRIO1CcAAAAJ)<sup>1</sup>, [Takashi Shibuya](https://scholar.google.com/citations?user=XCRO260AAAAJ)<sup>1</sup>, [Yuki Mitsufuji](https://www.yukimitsufuji.com/)<sup>1,2</sup>

<sup>1</sup>Sony AI and <sup>2</sup>Sony Group Corporation

## Installation

We recommend using a [Miniforge](https://github.com/conda-forge/miniforge) environment.

**1. Clone the repository**

```bash
git clone https://github.com/SonyResearch/MMDisCo.git
```

**2. Install prerequisites if needed**

```bash
conda install -c conda-forge ffmpeg
```

**3. Install required Python libraries**

```bash
cd MMDisCo
pip install -e .
```

## Downloading Pre-trained Weights

### Base Models

You need to download the weights of [MM-Diffusion](https://github.com/researchmm/MM-Diffusion) and [VideoCrafter2](https://github.com/AILab-CVC/VideoCrafter?tab=readme-ov-file#-models) if you want to use these models as the base models.
The weights of other models are automatically downloaded.

**MM-Diffusion**

Please follow the [official instructions](https://github.com/researchmm/MM-Diffusion?tab=readme-ov-file#models) to download the pre-trained models.
The weights for the base model (64x64) and the super-resolution model (64x64 -> 256x256) are needed, and the downloaded weights should be placed in the `./checkpoints/mmdiffusion` directory.

**VideoCrafter2**

Please follow the [official instructions](https://github.com/AILab-CVC/VideoCrafter?tab=readme-ov-file#1-text-to-video) to download the pre-trained Text-to-Video model.
The downloaded weights should be placed in the `./checkpoints/video_crafter` directory.

### MMDisCo

The models are available at [https://huggingface.co/AkioHayakawa/MMDisCo](https://huggingface.co/AkioHayakawa/MMDisCo).
Download all files and place them in the `./checkpoints/mmdisco` directory.

The expected directory structure that includes all pre-trained weights is:

```
MMDisCo
└── checkpoints
    ├── mmdiffusion
    │   ├── AIST++.pt
    │   ├── AIST++_SR.pt
    │   ├── landscape.pt
    │   └── landscape_SR.pt
    ├── video_crafter
    │   └── base_512_v2
    │       └── model.ckpt
    └── mmdisco
        ├── audioldm_animediff_vggsound.pt
        ├── auffusion_videocrafter2_vggsound.pt
        ├── mmdiffusion_aist++.pt
        └── mmdiffusion_landscape.pt
```

## Generating Video and Audio Using a Pre-trained Model

### Joint Audio and Video Generation Using Pre-trained Text-to-Audio and Text-to-Video Models with MMDisCo

We provide a demo script for joint audio and video generation using two pairs of base models: [AudioLDM](https://arxiv.org/abs/2301.12503) / [AnimateDiff](https://arxiv.org/abs/2307.04725) and [Auffusion](https://arxiv.org/abs/2401.01044) / [VideoCrafter2](https://arxiv.org/abs/2401.09047).

The generation script can be run as follows:

```bash
cd scripts/

# Using AudioLDM / AnimateDiff as base models
python generate.py model=audioldm_animediff_vggsound

# Using Auffusion / VideoCrafter2 as base models
python generate.py model=auffusion_videocrafter2_vggsound
```

The output videos will be placed in the `scripts/output/generate/` directory by default.

### MMDisCo as Joint Guidance for the Pre-trained Joint Generation Model

We provide a demo script for generating outputs from the pre-trained joint generation model enhanced by MMDisCo's joint guidance. We support [MM-Diffusion](https://arxiv.org/abs/2212.09478) as a base model. We provide MMDisCo for models trained on the AIST++ (dance music video) and Landscape (natural scene video) datasets.

The generation script can be run as follows:

```bash
cd scripts/

# Using the model trained on AIST++
# (Use double quotes "" for the model argument depending on your shell environment.)
python generate_mmdiffusion.py "model=mmdiffusion_aist++"

# Using the model trained on Landscape
python generate_mmdiffusion.py model=mmdiffusion_landscape
```

## Citation

```
@inproceedings{hayakawa2025mmdisco,
title={{MMD}is{C}o: Multi-Modal Discriminator-Guided Cooperative Diffusion for Joint Audio and Video Generation},
author={Akio Hayakawa and Masato Ishii and Takashi Shibuya and Yuki Mitsufuji},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025}
}
```
