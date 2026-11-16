"""Text-driven facial expression editing / reenactment on top of CVTHead.

Examples
--------
Face reenactment (source image driven by another face image)::

    python infer_seq.py --src_pth examples/s01.png --drv_pth examples/s02.png \
        --out_pth outputs/reenact.png

Text-driven expression editing (FLAME coefficients predicted from a caption)::

    python infer_seq.py --flame --src_pth examples/s01.png --out_pth outputs \
        --mode fet --dataset ck \
        --caption "Turn this face from anger to happiness"
"""
import argparse
import os
import time

import cv2
import face_alignment
import numpy as np
import torch
from PIL import Image
from skimage.transform import warp
from transformers import CLIPTextModel, CLIPTokenizer

from dataset.video_data import get_deca_tform
from models.cvthead import CVTHead
from models.fet import FET_MODEL
from models.motion import EXP_CVAE, POSE_CVAE, NEUTRAL_CVAE
from models.motionclip import MOTIONCLIP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

DATA_DIR = "./data"
LATENT_DIM = 16
N_POSE_DIM = 6
N_EXP_DIM = 50
N_FRAMES = 2
INTERMEDIATE_DIM = 128  # size of the hidden layer

DEFAULT_CAPTIONS = [
    "Modify this face, changing it from happiness to surprise",
    "Turn this face from anger to happiness",
    "Change this face from happiness to sadness",
    "Transform this face from sadness to fear",
    "Replace this face from sadness to surprise",
    "Turn this face from anger to sadness",
]


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #
def mask_erosion(mask):
    kernel = np.ones((9, 9), np.uint8)
    resmask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return torch.from_numpy(resmask / 255)


def tensor2image(tensor):
    image = tensor.detach().cpu().numpy()
    image = image * 255.
    image = np.maximum(np.minimum(image, 255), 0)
    image = image.transpose(1, 2, 0)[:, :, [2, 1, 0]]
    return image.astype(np.uint8).copy()


def preprocess_image(img_pth, fa, device=device):
    img = Image.open(img_pth).convert("RGB")
    img = img.resize((256, 256))

    img_npy = np.array(img)                     # (256, 256, 3), val: 0-255
    landmarks = fa.get_landmarks(img_npy)
    if not landmarks:
        raise RuntimeError(f"No face detected in {img_pth}")
    tform = get_deca_tform(landmarks[0])        # (3, 3)

    img_npy = img_npy / 255.                    # (H, W, 3), val: 0-1
    crop_image = warp(img_npy, tform.inverse, output_shape=(224, 224))   # (224, 224, 3), val: [0, 1]

    img_tensor = torch.from_numpy(img_npy).float().permute(2, 0, 1)     # (3, H, W), val: [0, 1]
    img_tensor = (img_tensor - 0.5) / 0.5                                # (3, H, W), val: [-1, 1]

    crop_image = torch.tensor(np.asarray(crop_image)).float().permute(2, 0, 1)  # (3, 224, 224), val: [0, 1]
    tform = torch.tensor(np.asarray(tform)).float()                             # (3, 3)

    return (img_tensor.unsqueeze(0).to(device),   # (1, 3, 256, 256)
            crop_image.unsqueeze(0).to(device),   # (1, 3, 224, 224)
            tform.unsqueeze(0).to(device))        # (1, 3, 3)


def postprocess(model, outputs, softmask):
    """Convert CVTHead outputs into a masked RGB PIL image on a white background."""
    predict_img = outputs["pred_drv_img"]     # (1, 3, 256, 256), val: [-1, 1]
    predict_mask = outputs["pred_drv_mask"]   # (1, 1, 256, 256), val: [0, 1], soft mask

    predict_img = 0.5 * (predict_img + 1)
    predict_img = predict_img[0].permute(1, 2, 0).cpu().numpy()     # (256, 256, 3)
    predict_mask = predict_mask[0].permute(1, 2, 0).cpu().numpy()   # (256, 256, 1)

    if not softmask:
        predict_mask = (predict_mask > 0.6) + 0.0
        predict_mask = mask_erosion(predict_mask * 255).unsqueeze(2).cpu().numpy()

    # apply predicted mask
    predict_img = predict_img * predict_mask + (1 - predict_mask)

    # refine with face parsing
    parse_in = torch.from_numpy(predict_img).unsqueeze(0).permute(0, 3, 1, 2).float().to(device)
    parsing, _ = model.face_parsing(parse_in)
    face_mask = (parsing.argmax(1) > 0).float()               # (1, H, W)
    face_mask = face_mask.squeeze(0).unsqueeze(2).cpu().numpy()
    predict_img = predict_img * face_mask + (1 - face_mask)

    predict_img = (predict_img * 255).astype(np.uint8)
    return Image.fromarray(predict_img)


def interpolate(model, src_img_crop, src_img, src_tform, exp_a, pose_a, exp_b, pose_b,
                softmask, step=0.1):
    """Render frames while linearly interpolating FLAME (exp, pose) from a to b."""
    frames = []
    for alpha in np.arange(0, 1 + step, step):
        exp = exp_a * (1 - alpha) + alpha * exp_b
        pose = pose_a * (1 - alpha) + alpha * pose_b
        outputs = model.flame_coef_generation(src_img_crop, src_img, src_tform, exp=exp, pose=pose)
        frames.append(postprocess(model, outputs, softmask))
    return frames


# --------------------------------------------------------------------------- #
# Driving modes
# --------------------------------------------------------------------------- #
def driven_by_face(model, src_pth, drv_pth, out_pth, softmask=True):
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(device))

    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa)
    drv_img, drv_img_crop, drv_tform = preprocess_image(drv_pth, fa)

    with torch.no_grad():
        outputs = model(src_img_crop, drv_img_crop, src_img, drv_img, src_tform, drv_tform,
                        is_train=False, is_cross_id=True)
        predict_img = 0.5 * (outputs["pred_drv_img"] + 1)
        predict_img = predict_img[0].permute(1, 2, 0).cpu().numpy()
        predict_mask = outputs["pred_drv_mask"][0].permute(1, 2, 0).cpu().numpy()
        if not softmask:
            predict_mask = (predict_mask > 0.6).astype(np.float32)

        predict_img = predict_img * predict_mask + (1 - predict_mask)
        predict_img = Image.fromarray((predict_img * 255).astype(np.uint8))

    if os.path.splitext(out_pth)[1] == "":
        os.makedirs(out_pth, exist_ok=True)
        out_pth = os.path.join(out_pth, "reenact.png")
    else:
        os.makedirs(os.path.dirname(out_pth) or ".", exist_ok=True)
    predict_img.save(out_pth)
    print(f"Saved {out_pth}")


def driven_by_text(model, src_pth, out_pth, captions, mode, dataset_name,
                   use_neutral=True, softmask=False):
    """Edit the source face according to text instructions.

    With ``use_neutral`` the animation goes source -> expression A -> neutral -> expression B,
    otherwise source -> expression A -> expression B.
    """
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=str(device))
    src_img, src_img_crop, src_tform = preprocess_image(src_pth, fa)

    motion_models = define_model(mode, dataset_name)

    start = time.time()
    for caption in captions:
        save_path = os.path.join(out_pth, caption.replace(" ", "_"))
        os.makedirs(save_path, exist_ok=True)

        if mode in ("cvae", "motionclip"):
            cvae, cvae_pose, cvae_neutral, tokenizer, text_encoder = motion_models
            pose1, pose2, exp1, exp2, neutral_exp, neutral_pose = run_samples(
                cvae_pose, cvae, cvae_neutral, tokenizer, text_encoder, dataset_name, caption, mode)
        else:  # fet
            fet, cvae_neutral, tokenizer, text_encoder = motion_models
            pose1, pose2, exp1, exp2, neutral_exp, neutral_pose = run_fet_samples(
                fet, cvae_neutral, tokenizer, text_encoder, dataset_name, caption)

        with torch.no_grad():
            exp1, pose1 = torch.from_numpy(exp1).to(device), torch.from_numpy(pose1).to(device)
            exp2, pose2 = torch.from_numpy(exp2).to(device), torch.from_numpy(pose2).to(device)
            n_exp, n_pose = torch.from_numpy(neutral_exp).to(device), torch.from_numpy(neutral_pose).to(device)

            code_init = model.flame_coef_init_generation(src_img_crop, src_img)
            keyframes = [(code_init["exp"], code_init["pose"]), (exp1, pose1)]
            if use_neutral:
                keyframes.append((n_exp, n_pose))
            keyframes.append((exp2, pose2))

            frames = []
            for seg_id, ((exp_a, pose_a), (exp_b, pose_b)) in enumerate(zip(keyframes[:-1], keyframes[1:]), start=1):
                seg_frames = interpolate(model, src_img_crop, src_img, src_tform,
                                         exp_a, pose_a, exp_b, pose_b, softmask)
                seg_frames[-1].save(os.path.join(save_path, f"rgb_{seg_id}.jpg"))
                frames.extend(seg_frames)

            out_name = os.path.join(save_path, f"{caption.replace(' ', '_')}.gif")
            frames[0].save(out_name, format="GIF", append_images=frames, save_all=True,
                           duration=100, loop=0)
            print(f"Saved {out_name}")

    print(f"Time: {time.time() - start:.2f}s")


# --------------------------------------------------------------------------- #
# Text-to-motion models
# --------------------------------------------------------------------------- #
def define_model(mode, dataset_name):
    pretrained_model_path = os.path.join(DATA_DIR, "StableDiffusion")
    torch.manual_seed(1)

    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    text_encoder.requires_grad_(False)
    text_encoder.to(device)
    cvae_neutral = NEUTRAL_CVAE(intermediate_dim=256, latent_dim=16, n_exp=1, exp_dim=50,
                                pose_dim=6, cam_dim=3).to(device)

    if mode == "cvae":
        cvae = EXP_CVAE(INTERMEDIATE_DIM, LATENT_DIM, n_exp=N_FRAMES, n_dim=N_EXP_DIM).to(device)
        cvae_pose = POSE_CVAE(INTERMEDIATE_DIM, LATENT_DIM, n_exp=N_FRAMES, n_dim=N_POSE_DIM).to(device)
        return cvae, cvae_pose, cvae_neutral, tokenizer, text_encoder

    if mode == "motionclip":
        cvae = MOTIONCLIP(LATENT_DIM, N_FRAMES, N_EXP_DIM, N_POSE_DIM, device=device,
                          text_encoder=text_encoder, tokenizer=tokenizer, mmode="exp").to(device)
        cvae.load_state_dict(_load(f"{dataset_name}_exp_motionclip0.pth")["state_dict"])

        cvae_pose = MOTIONCLIP(LATENT_DIM, N_FRAMES, N_EXP_DIM, N_POSE_DIM, device=device,
                               text_encoder=text_encoder, tokenizer=tokenizer, mmode="pose").to(device)
        cvae_pose.load_state_dict(_load(f"{dataset_name}_pose_motionclip0.pth")["state_dict"])
        return cvae, cvae_pose, cvae_neutral, tokenizer, text_encoder

    if mode == "fet":
        fet = FET_MODEL(INTERMEDIATE_DIM, LATENT_DIM, n_exp=N_FRAMES, device=device).to(device)
        fet.load_state_dict(_load(f"{dataset_name}_av01.pth")["state_dict"], strict=False)
        return fet, cvae_neutral, tokenizer, text_encoder

    raise ValueError(f"Unknown mode: {mode}")


def _load(name):
    return torch.load(os.path.join(DATA_DIR, name), map_location=device)


def lengths_to_mask(lengths):
    max_len = max(lengths)
    return torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)


def encode_text(tokenizer, text_encoder, caption):
    prompt_ids = tokenizer(caption, max_length=tokenizer.model_max_length, padding="max_length",
                           truncation=True, return_tensors="pt").input_ids.to(device)
    return text_encoder(prompt_ids)[0]


def inference(m_cvae, tokenizer, text_encoder, dataset_name, caption, nn_dim=50, mode="exp",
              latent_dim=LATENT_DIM):
    m_cvae.load_state_dict(_load(f"{dataset_name}_{mode}_cvae.pth")["state_dict"])

    sample = np.random.normal(0, 1, size=(1, latent_dim))
    inp_sample = torch.tensor(sample, dtype=torch.float, device=device)
    encoder_hidden_states = m_cvae.emb(encode_text(tokenizer, text_encoder, caption))
    out = m_cvae.decode(inp_sample, encoder_hidden_states).detach().cpu().numpy()
    return out[:, :nn_dim], out[:, nn_dim:2 * nn_dim]


def motionclip_inference(m_cvae, tokenizer, text_encoder, caption, nn_dim=50):
    n_samples, n_exp = 1, 2
    sample = np.random.normal(0, 1, size=(n_samples, nn_dim))
    inp_sample = torch.tensor(sample, dtype=torch.float, device=device)

    encoder_hidden_states = m_cvae.textemb(encode_text(tokenizer, text_encoder, caption))
    mask = lengths_to_mask(torch.tensor([n_exp] * n_samples))
    out = m_cvae.decode(inp_sample, encoder_hidden_states, mask)["output"]
    out = out.detach().cpu().numpy().reshape((n_samples, n_exp, nn_dim))
    return out[:, 0, :], out[:, 1, :]


def fet_inference(fet, tokenizer, text_encoder, caption):
    n_samples, n_exp = 1, 2
    out_exp, out_pose = fet.decode(encode_text(tokenizer, text_encoder, caption))
    out_exp = out_exp.detach().cpu().numpy().reshape((n_samples, n_exp, N_EXP_DIM))
    out_pose = out_pose.detach().cpu().numpy().reshape((n_samples, n_exp, N_POSE_DIM))
    return out_exp[:, 0, :], out_exp[:, 1, :], out_pose[:, 0, :], out_pose[:, 1, :]


def inference_neutral(m_cvae, dataset_name, latent_dim=LATENT_DIM):
    sample = np.random.normal(0, 1, size=(1, latent_dim))
    inp_sample = torch.tensor(sample, dtype=torch.float, device=device)
    m_cvae.load_state_dict(_load(f"{dataset_name}_neutral_cvae.pth")["state_dict"])
    pexp, ppose = m_cvae.decode(inp_sample)
    return pexp.detach().cpu().numpy(), ppose.detach().cpu().numpy()


def run_samples(cvae_pose, cvae, cvae_neutral, tokenizer, text_encoder, dataset_name, caption, mode):
    if mode == "cvae":
        pose1, pose2 = inference(cvae_pose, tokenizer, text_encoder, dataset_name, caption,
                                 nn_dim=N_POSE_DIM, mode="pose")
        exp1, exp2 = inference(cvae, tokenizer, text_encoder, dataset_name, caption,
                               nn_dim=N_EXP_DIM, mode="exp")
    else:  # motionclip
        pose1, pose2 = motionclip_inference(cvae_pose, tokenizer, text_encoder, caption, nn_dim=N_POSE_DIM)
        exp1, exp2 = motionclip_inference(cvae, tokenizer, text_encoder, caption, nn_dim=N_EXP_DIM)
    neutral_exp, neutral_pose = inference_neutral(cvae_neutral, dataset_name)
    return pose1, pose2, exp1, exp2, neutral_exp, neutral_pose


def run_fet_samples(fet, cvae_neutral, tokenizer, text_encoder, dataset_name, caption):
    exp1, exp2, pose1, pose2 = fet_inference(fet, tokenizer, text_encoder, caption)
    neutral_exp, neutral_pose = inference_neutral(cvae_neutral, dataset_name)
    return pose1, pose2, exp1, exp2, neutral_exp, neutral_pose


# --------------------------------------------------------------------------- #
def main(args):
    model = CVTHead()
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(device)

    print(f"Loading {args.ckpt_pth}")
    ckpt = torch.load(args.ckpt_pth, map_location="cpu", encoding="latin1")["model"]
    model.load_state_dict(ckpt, strict=False)
    print(f"-- Number of parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M\n")

    if args.flame:
        driven_by_text(model, args.src_pth, args.out_pth, args.caption or DEFAULT_CAPTIONS,
                       mode=args.mode, dataset_name=args.dataset,
                       use_neutral=not args.no_neutral, softmask=False)
    else:
        driven_by_face(model, args.src_pth, args.drv_pth, args.out_pth, softmask=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CVTHead inference")
    parser.add_argument("--src_pth", type=str, default="examples/s01.png", help="Source face image.")
    parser.add_argument("--drv_pth", type=str, default="examples/s02.png",
                        help="Driving face image (face reenactment mode only).")
    parser.add_argument("--out_pth", type=str, default="outputs",
                        help="Output directory (or output image path in reenactment mode).")
    parser.add_argument("--ckpt_pth", type=str, default="data/cvthead.pth", help="CVTHead checkpoint.")
    parser.add_argument("--flame", action="store_true",
                        help="Text-driven editing via predicted FLAME coefficients.")
    parser.add_argument("--mode", type=str, default="fet", choices=["fet", "cvae", "motionclip"],
                        help="Text-to-motion model used in --flame mode.")
    parser.add_argument("--dataset", type=str, default="ck", choices=["ck", "celebv"],
                        help="Dataset the text-to-motion checkpoints were trained on.")
    parser.add_argument("--caption", type=str, nargs="+", default=None,
                        help="One or more text instructions (default: built-in examples).")
    parser.add_argument("--no_neutral", action="store_true",
                        help="Skip the intermediate neutral expression in the animation.")
    main(parser.parse_args())
