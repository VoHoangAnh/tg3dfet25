"""Evaluate text-driven facial expression editing (L1 / PSNR / SSIM / MS-SSIM / LPIPS / FID).

Example::

    python eval.py --config configs/eval_ck.yaml --mode motionclip
"""
import argparse
import logging
import os

import torch
import yaml

import dataset
from dataset.mt_data import MTDataset
from models.cvthead import CVTHead
from utils.common import init_ddp
from utils.logger import set_logger
from utils.trainer import Trainer

def main(args):
    # >>>>>>>>>>>>>>>>> All Hyper Parameters >>>>>>>>>>>>>>>>>
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    out_dir = cfg["exp"]
    os.makedirs(out_dir, exist_ok=True)
    set_logger(os.path.join(out_dir, 'eval_celebv.log'))

    # >>>>>>>>>>>>>>>>> Distributed Data Parallel set up >>>>>>>>>>>>>>>>>
    rank, world_size = init_ddp()
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

    # >>>>>>>>>>>>>>>>> Initialize datasets >>>>>>>>>>>>>>>>>
    data_path = cfg["data"]["path"]
    meta_path = cfg["data"]["meta"]
    json_path = cfg["data"]["json"]
    dataset_name = cfg["data"]["dataset_name"]
    ckpt_pth = cfg["model"]["ckpt_pth"]

    print(f"world_size: {world_size}")
    batch_size = 1
    print(f"batch_size: {batch_size}")

    eval_dataset = MTDataset(root_dir=data_path, meta_dir=meta_path, jsonfile = json_path, dataset_name=dataset_name, id_sampling=False, is_train=False)
    logging.info("--- Total eval {}".format(len(eval_dataset)) )

    num_workers = cfg['training']['num_workers'] if 'num_workers' in cfg['training'] else 1
    logging.info(f'--- Using {num_workers} workers per process for data loading.')

    val_loader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False,
        pin_memory=False, worker_init_fn=dataset.worker_init_fn,
        persistent_workers=num_workers > 0)

    # >>>>>>>>>>>>>>>>> Model >>>>>>>>>>>>>>>>>
    model = CVTHead().to(device)
    ckpt = torch.load(ckpt_pth, map_location="cpu", encoding='latin1')["model"]
    model.load_state_dict(ckpt, strict=False)
    print(f'-- Number of parameters (G): {sum(p.numel() for p in model.parameters())/1e6} M\n')

    logging.info("[main] Load pre-trained CVTHead...")

    generator = None 
    discriminator = None 
    optimizer_G = None 
    optimizer_D = None 
    
    trainer = Trainer(model, generator, discriminator, optimizer_G, optimizer_D, cfg, device, out_dir, stage=2)

    logging.info(f"\n--- Evaluation (mode={args.mode}) ---")
    eval_dict = trainer.evaluate_m2t(val_loader, 0, dataset_name, args.mode)
    print(eval_dict)


if __name__ == '__main__':
    #>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Arguments >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    parser = argparse.ArgumentParser(description='Evaluate text-driven expression editing.')
    parser.add_argument('--config', type=str, required=True, help='Path to config file.')
    parser.add_argument('--mode', type=str, default='motionclip', choices=['cvae', 'fet', 'motionclip'],
                        help='Text-to-motion model to evaluate.')

    args = parser.parse_args()

    main(args)
