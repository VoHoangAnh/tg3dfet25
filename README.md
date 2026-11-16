# Instruction-Driven 3D Facial Expression Generation and Transition

**Anh H. Vo, Tae-Seok Kim, Hulin Jin, Soo-Mi Choi, Yong-Guk Kim\***

*IEEE Transactions on Multimedia, 2025*

[**Project Page**](https://vohoanganh.github.io/tg3dfet/) &nbsp;|&nbsp;
[**Paper**](https://arxiv.org/abs/2601.08179)

<p align="center">
  <img src="https://vohoanganh.github.io/tg3dfet/static/images/problem.png" width="800"/>
</p>

This repository contains the rendering and evaluation code for the paper. Given a single face image
and a natural-language instruction such as *"Turn this face from anger to happiness"*, a
text-to-motion model predicts FLAME expression and pose coefficients, and a
[CVTHead](https://github.com/HowieMa/CVTHead)-based neural renderer animates the source face
through the transition, producing a GIF.

> **Note:** The rendering module is adapted from
> [**CVTHead**](https://github.com/HowieMa/CVTHead) — *"CVTHead: One-shot Controllable Head Avatar
> with Vertex-feature Transformer"* (WACV 2024). Many thanks to the authors for releasing their code.
> If you use the renderer, please also cite their paper).

Sample inputs:

<p align="center">
  <img src="examples/s01.png" width="160"/>
  <img src="examples/s02.png" width="160"/>
  <img src="examples/s03.png" width="160"/>
</p>

## Features

- **Text-driven editing**: three interchangeable text-to-motion back-ends
  - `fet` — transformer-based FLAME expression model (`models/fet.py`)
  - `cvae` — conditional VAEs for expression and pose (`models/motion.py`)
  - `motionclip` — MotionCLIP-style model (`models/motionclip.py`)
- Optional **neutral expression** prior between the two target expressions
- **Face reenactment**: drive a source face with another face image
- **Evaluation** on CK+ / CelebV-HQ: L1, PSNR, SSIM, MS-SSIM, LPIPS and FID

## Repository structure

```
.
├── infer_seq.py        # inference: text-driven editing and face reenactment
├── eval.py             # quantitative evaluation
├── configs/            # evaluation configs
├── dataset/            # dataset loaders (CK+, CelebV-HQ, VoxCeleb-style videos)
├── decalib/            # DECA / FLAME (3D face reconstruction)
├── losses/             # perceptual, identity and segmentation losses
├── models/             # CVTHead renderer and text-to-motion models
├── utils/              # trainer, metrics, logging, visualisation
├── examples/           # sample input images
└── data/               # pretrained weights and assets (not tracked, see data/README.md)
```

## Installation

Tested with Python 3.9 and a CUDA GPU.

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

conda create -n texpr python=3.9 -y
conda activate texpr

# 1. PyTorch (pick the command for your CUDA version at https://pytorch.org)
pip install torch torchvision

# 2. PyTorch3D and torch-scatter (must match your torch / CUDA build)
#    https://github.com/facebookresearch/pytorch3d/blob/main/INSTALL.md
#    https://github.com/rusty1s/pytorch_scatter#installation

# 3. Remaining dependencies
pip install -r requirements.txt
```

Optional: DECA's faster CUDA rasterizer can be built with
`cd decalib/utils/rasterizer && python setup.py build_ext -i` (PyTorch3D is used by default).

## Pretrained models

Download the pretrained weights and FLAME/DECA assets into `data/`.
See **[data/README.md](data/README.md)** for the full list and sources.

> FLAME requires registration at <https://flame.is.tue.mpg.de/> and is distributed under its own license.

## Usage

### Text-driven expression editing

```bash
python infer_seq.py --flame \
    --src_pth examples/s01.png \
    --out_pth outputs \
    --mode fet --dataset ck \
    --caption "Turn this face from anger to happiness" "Change this face from happiness to sadness"
```

For each caption a folder `outputs/<caption>/` is created containing the key frames
(`rgb_1.jpg`, `rgb_2.jpg`, …) and an animated GIF of the full transition.

| Argument | Default | Description |
|---|---|---|
| `--src_pth` | `examples/s01.png` | Source face image |
| `--out_pth` | `outputs` | Output directory |
| `--ckpt_pth` | `data/cvthead.pth` | CVTHead checkpoint |
| `--mode` | `fet` | Text-to-motion model: `fet`, `cvae`, `motionclip` |
| `--dataset` | `ck` | Dataset the text-to-motion checkpoints were trained on: `ck`, `celebv` |
| `--caption` | built-in examples | One or more instructions |
| `--no_neutral` | off | Go directly from expression A to B, without the neutral face in between |

### Face reenactment

```bash
python infer_seq.py --src_pth examples/s01.png --drv_pth examples/s02.png --out_pth outputs/reenact.png
```

### Evaluation

1. Copy `configs/eval_ck.yaml` and set the dataset paths.
2. Run:

```bash
python eval.py --config configs/eval_ck.yaml --mode motionclip
```

The evaluation JSON lists image pairs with their captions:

```json
[
  {
    "caption": ["Turn this face from anger to happiness"],
    "lb_target1": "AN", "img_target1": "S010_004_00000017.png",
    "lb_target2": "HA", "img_target2": "S010_006_00000015.png"
  }
]
```

(CelebV-HQ entries also need a `subj` field.) Labels use the codes
`AN, CONT, DI, FE, HA, NE, SA, SU`.

## Acknowledgements

This code builds on the following projects; please also respect their licenses:

- [CVTHead](https://github.com/HowieMa/CVTHead) — *"CVTHead: One-shot Controllable Head Avatar with Vertex-feature Transformer"* (WACV 2024); our rendering module is based on this work
- [DECA](https://github.com/yfeng95/DECA) and [FLAME](https://flame.is.tue.mpg.de/) — 3D face model and reconstruction
- [ROME](https://github.com/SamsungLabs/rome) — mesh deformation components
- [MotionCLIP](https://github.com/GuyTevet/MotionCLIP) — text–motion alignment
- [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) — face segmentation
- [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) — identity features
- [Hugging Face Transformers](https://github.com/huggingface/transformers) — CLIP text encoder

## Citation

If you find this work useful, please cite:

```bibtex
@article{vo2025tmm,
  author  = {Vo, Anh H. and Kim, Tae-Seok and Jin, Hulin and Choi, Soo-Mi and Kim, Yong-Guk},
  title   = {Instruction-Driven 3D Facial Expression Generation and Transition},
  journal = {IEEE Transactions on Multimedia},
  year    = {2025}
}
```

## License

TODO — choose a license (see the note on third-party licenses above). DECA and FLAME are
released for non-commercial research use only.
