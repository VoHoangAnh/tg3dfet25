# `data/` — pretrained weights and assets

Nothing in this folder is tracked by git except this file. Download / copy the files below
into `data/` before running anything. All paths are relative to the repository root.

## Renderer (CVTHead)

| File | Used by | Source |
|---|---|---|
| `cvthead.pth` | `infer_seq.py`, `eval.py` (`--ckpt_pth`) | [CVTHead](https://github.com/HowieMa/CVTHead) release |
| `79999_iter.pth` | face parsing (`models/face_parsing.py`) | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) |
| `linear_hair.pth`, `mean_deform.pt`, `u_full.pt`, `transform.pkl`, `rome.pth` (optional) | `models/cvthead.py` | [CVTHead](https://github.com/HowieMa/CVTHead) / [ROME](https://github.com/SamsungLabs/rome) |
| `resnet50_scratch_weight.pth` | identity loss (`utils/trainer.py`) | [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) |

## DECA / FLAME (`decalib/utils/config.py`)

| File | Source |
|---|---|
| `deca_model.tar` | [DECA](https://github.com/yfeng95/DECA) |
| `generic_model.pkl` | [FLAME 2020](https://flame.is.tue.mpg.de/) (registration required) |
| `head_template.obj`, `landmark_embedding.npy`, `fixed_displacement_256.npy`, `uv_face_mask.png`, `uv_face_eye_mask.png`, `mean_texture.jpg` | [DECA `data/`](https://github.com/yfeng95/DECA/tree/master/data) |
| `texture_data_256.npy` | [DECA](https://github.com/yfeng95/DECA) |
| `resnet50_ft_weight.pkl` | [VGGFace2-pytorch](https://github.com/cydonia999/VGGFace2-pytorch) |

## Text encoder

| Path | Source |
|---|---|
| `StableDiffusion/tokenizer/`, `StableDiffusion/text_encoder/` | CLIP tokenizer and text encoder from a Stable Diffusion v1.x checkpoint on the Hugging Face Hub (e.g. the `tokenizer` and `text_encoder` subfolders of `stable-diffusion-v1-5`) |

## Text-to-motion checkpoints

`<dataset>` is `ck` or `celebv`.

| File | Mode |
|---|---|
| `<dataset>_neutral_cvae.pth` | all modes (neutral expression prior) |
| `<dataset>_exp_cvae.pth`, `<dataset>_pose_cvae.pth` | `--mode cvae` |
| `<dataset>_exp_motionclip0.pth`, `<dataset>_pose_motionclip0.pth` | `--mode motionclip` |
| `<dataset>_av01.pth` | `--mode fet` |

<!-- TODO: add download links for these checkpoints once they are released. -->

Expected layout:

```
data/
├── cvthead.pth
├── 79999_iter.pth
├── deca_model.tar
├── generic_model.pkl
├── ...
├── ck_av01.pth
├── ck_neutral_cvae.pth
└── StableDiffusion/
    ├── tokenizer/
    └── text_encoder/
```
