import torch 
import numpy as np
import torchvision
import os 
from torch.utils.data.dataset import Dataset
import json
from transformers import CLIPTextModel, CLIPTokenizer
import torch.nn as nn
from collections import OrderedDict
import time 
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import os 
import imageio 
from PIL import Image
import torchvision.transforms as transforms
from torchvision.models import resnet101, ResNet101_Weights
#from torchmetrics import Accuracy

from scipy.io import savemat
#from DECA.decalib.deca import DECA
#from DECA.decalib.datasets import datasets 
#from DECA.decalib.utils import util
#from DECA.decalib.utils.config import cfg as deca_cfg
#from DECA.decalib.utils.tensor_cropper import transform_points
import itertools 
import matplotlib.pyplot as plt 
#from sklearn.metrics import confusion_matrix
import cv2 
#from imblearn.metrics import geometric_mean_score


loss_mse = nn.MSELoss(reduction = "mean")
loss_ce = nn.CrossEntropyLoss()
cosine_sim = nn.CosineSimilarity(dim=1, eps=1e-6)
################## MOTIONCLIP ####################    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
       
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x): ## [3, 100]
        # not used in the final model
        x = x + self.pe[:x.shape[0], :]
        return self.dropout(x)

## only for ablation / not used in the final model ## 
class TimeEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, mask, lengths):
        time = mask * 1/(lengths[..., None]-1)
        time = time[:, None] * torch.arange(time.shape[1], device=x.device)[None, :]
        time = time[:, 0].T
        # add the time encoding
        x = x + time[..., None]
        return self.dropout(x)
    
class TimestepEmbedder(nn.Module):
    def __init__(self, latent_dim, sequence_pos_encoder):
        super().__init__()
        self.latent_dim = latent_dim
        self.sequence_pos_encoder = sequence_pos_encoder

        time_embed_dim = self.latent_dim
        self.time_embed = nn.Sequential(
            nn.Linear(self.latent_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

    def forward(self, timesteps):
        return self.time_embed(self.sequence_pos_encoder.pe[timesteps]).permute(1, 0, 2)
    
class Encoder_TRANSFORMER(nn.Module):
    def __init__(self, n_exp, exp_dim, pose_dim, device,
                latent_dim=128, ff_size=128, num_layers=2, num_heads=2, dropout=0.1,
                ablation=None, activation="gelu", mmode='exp', **kargs):
        
        super().__init__()
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.activation = activation
        self.ablation = ablation
        self.n_exp = n_exp
        self.exp_dim = exp_dim 
        self.pose_dim = pose_dim
        self.device = device 
        bs = 128 
        if mmode=="exp":
            self.input_feats = exp_dim
        elif mmode=="pose":
            self.input_feats = pose_dim

        self.mu_layer = nn.Linear(self.n_exp*self.latent_dim, self.latent_dim)
        self.sigma_layer = nn.Linear(self.n_exp*self.latent_dim, self.latent_dim)
       
        # self.sequence_pos_encoder = PositionalEncoding(2*self.latent_dim, self.dropout)
        # n_dmodel = self.num_heads*self.latent_dim
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)
        n_dmodel = self.latent_dim
        
        seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=n_dmodel, #self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=self.activation)
        
        self.seqTransEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                     num_layers=self.num_layers)

        self.skelEmbedding = nn.Linear(self.input_feats, self.latent_dim)
        
        self.muQuery = nn.Parameter(torch.randn(1, self.latent_dim))
        self.sigmaQuery = nn.Parameter(torch.randn(1, self.latent_dim))
       
        self.layer_norm = nn.LayerNorm(self.latent_dim, eps=1e-6)

    def forward(self, x, text_emb, mask):
        # x, y, mask = batch["x"], batch["y"], batch["mask"]
        # bs, njoints, nfeats, nframes = x.shape
        # bs, nfeats = x.shape ## [1, 100]
         
        ## embedding of the skeleton ##
        x = self.skelEmbedding(x)
        if self.ablation == "average_encoder":
            x = self.sequence_pos_encoder(x)
            ## transformer layers ##
            final = self.seqTransEncoder(x, src_key_padding_mask=None)

            ## get the average of the output ##
            z = final.mean(axis=0)
            
            ## extract mu and logvar ##
            mu = self.mu_layer(z)
            logvar = self.sigma_layer(z)
        
        else:
            ## adding the mu and sigma queries ##
            # text_emb = text_emb.unsqueeze(1)
            y = 0 # text_emb - text_emb 
          
            # print(x.shape)
           
            k1 = self.muQuery[y][None] ## [1, latent_dim]
            k2 = self.sigmaQuery[y][None] ## [1, latent_dim]
            k1 = k1.repeat(128,1).unsqueeze(1) ## [bs, 1, latent_dim]
            k2 = k2.repeat(128,1).unsqueeze(1) ## [bs, 1, latent_dim]

            # print(k1.shape)
            # print(k2.shape)
            # print(x.shape)
            xseq = torch.cat((k1, k2, x), axis=1)
            # xseq = torch.cat((x, text_emb), axis=1) ## [2, 100]
            
            ## add positional encoding ##
            xseq = self.sequence_pos_encoder(xseq) ## [2, 2, 100] if neutral: [3,3,100]
            muandsigmaMask = torch.ones((xseq.shape[0], 2), dtype=bool, device=x.device)

            maskseq = torch.cat((muandsigmaMask, mask.to(self.device)), axis=1) # [128, 3]
            maskseq = maskseq.permute(1,0).to(self.device)
            
            ## create a bigger mask, to allow attend to mu and sigma ##
            final = self.seqTransEncoder(xseq, src_key_padding_mask=~maskseq) ## [2, 2, 100]

            ## get the average of the output ##
            ## z = final.mean(axis=0) ## [2, 100]
            ## extract mu and logvar ##
            ## mu = self.mu_layer(z) ## [2, 100]
            ## logvar = self.sigma_layer(z) ## [2, 100]

            final = self.layer_norm(final)
            final = final.permute(1,0,2)
            mu = final[0,:,:] ## [2, n_dim]
            logvar = final[1,:,:] ## [2, n_dim]

            # print(final.shape)
            # print(mu.shape)
            # print(logvar.shape)

        return {"mu": mu, "logvar": logvar}

class Decoder_TRANSFORMER(nn.Module):
    def __init__(self, n_exp, exp_dim, pose_dim, device,
                 latent_dim=128, ff_size=128, num_layers=2, num_heads=2, dropout=0.1, activation="gelu",
                 ablation=None, mmode='exp', **kargs):
        super().__init__()
        
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.latent_dim = latent_dim 
        self.ff_size = ff_size
        self.device = device 
        if mmode=='exp':
            self.input_feats = exp_dim
        elif mmode=='pose':
            self.input_feats = pose_dim

        self.activation = activation
        self.ablation = ablation
        
        self.n_exp = n_exp
        self.exp_dim = exp_dim
        self.pose_dim = pose_dim

        ### only for ablation / not used in the final model ###
        if self.ablation == "time_encoding":
            self.sequence_pos_encoder = TimeEncoding(self.dropout)
        else:
            # self.sequence_pos_encoder = PositionalEncoding(2*self.latent_dim, self.dropout)
            self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # n_dmodel = self.n_exp*self.latent_dim
        n_dmodel = self.latent_dim
        seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=n_dmodel, #self.latent_dim,
                                                          nhead=self.num_heads,
                                                          dim_feedforward=self.ff_size,
                                                          dropout=self.dropout,
                                                          activation=activation)
        
        self.seqTransDecoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                     num_layers=self.num_layers)
        
        self.layer_norm = nn.LayerNorm(self.latent_dim, eps=1e-6)
        
        # if self.is_neutral:
        #     self.finallayer = nn.Linear(2*self.latent_dim, self.input_feats) ## [100, 100]
        # else:
        self.finallayer = nn.Linear(self.latent_dim, self.input_feats) ## [100, 100]


    def forward(self, z, text_emb, mask):
        z = z.unsqueeze(0) ## [1, 2, 100]
        bs = z.shape[1]
        lengths = self.n_exp
        # mask = None ##
       
        ## only for ablation / not used in the final model ##
        text_emb = text_emb.unsqueeze(0) 
        # z = torch.cat((z, text_emb), axis=0) ## [1, bs, 4, dim]
        z = text_emb
       
        timequeries = torch.zeros(self.n_exp, bs, self.latent_dim, device=z.device) ## [n_frame, bs, latent]
        # print(timequeries.shape) #
        ## only for ablation / not used in the final model ## 
        if self.ablation == "time_encoding":
            timequeries = self.sequence_pos_encoder(timequeries, mask, lengths)
        else:
            timequeries = self.sequence_pos_encoder(timequeries) ## [N, S, E]
        
        # z = z.squeeze(0)
        # print(z.shape)
        # timequeries=timequeries.permute(1,0,2) # [bs, n_frame, n_dim]
        # mask = mask.permute(1,0).to(device) ## [n_frame, bs]
        mask = mask.to(self.device)
       
        # print(timequeries.shape)
        # print(z.shape)
        # print(mask.shape)
        output = self.seqTransDecoder(tgt=timequeries, memory=z,
                                      tgt_key_padding_mask=~mask)
        
        output = self.layer_norm(output)
        output = self.finallayer(output)
        # output = output.mean(axis=1)
        output[~mask.T] = 0
        output = output.permute(1,0,2)       
        return {"output": output} 

class MOTIONCLIP(nn.Module):
    def __init__(self, latent_dim, n_exp, exp_dim, pose_dim, device, mmode='exp', **kwargs):
        super().__init__()
        self.latent_dim = latent_dim
        self.device = device 
        self.n_exp = n_exp 
        self.exp_dim = exp_dim 
        self.pose_dim = pose_dim 
        # self.tokenizer = tokenizer
        # self.text_encoder = text_encoder 
        self._matching_ = {"rc": self.compute_rc_loss, "kl": self.compute_kl_loss, "clip": self.compute_clip_losses}
        self.encoder = Encoder_TRANSFORMER(latent_dim=self.latent_dim, n_exp=self.n_exp, exp_dim=self.exp_dim, pose_dim=self.pose_dim, device = self.device, mmode=mmode).to(device)
        self.decoder = Decoder_TRANSFORMER(latent_dim=self.latent_dim, n_exp=self.n_exp, exp_dim=self.exp_dim, pose_dim=self.pose_dim, device = self.device, mmode=mmode).to(device)
       
        self.project = nn.Linear(59136, self.latent_dim).to(self.device) ## [bs, n_dim]
        self.proj_imb = nn.Linear(n_exp*512, self.latent_dim).to(self.device)
        # self.clip_model = kwargs['clip_model']
        # self.clip_lambdas = {'text': {"cosine":0.0}} ## [image, text]
        self.clip_lambdas = {'text': {"cosine":0.0}, 'image': {"cosine":0.0}} ## [image, text]
        self.clip_training = kwargs.get('clip_training', False)
        self.clip_model = None 

        if self.clip_training and self.clip_model:
            self.clip_model.training = True
        else:
            if self.clip_model:
                assert self.clip_model.training == False  # make sure clip is frozen

    @staticmethod
    def lengths_to_mask(lengths):
        max_len = max(lengths)
        if isinstance(max_len, torch.Tensor):
            max_len = max_len.item()
        index = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len)
        mask = index < lengths.unsqueeze(1)
        return mask
    
    def reparameterize(self, batch, seed=None):
        mu, logvar = batch["mu"], batch["logvar"] ## [bs, n_dim]
        std = torch.exp(logvar / 2) ## [bs, n_dim]

        if seed is None:
            eps = std.data.new(std.size()).normal_()
        else:
            generator = torch.Generator(device=self.device) ## function to set random value 
            generator.manual_seed(seed)
            eps = std.data.new(std.size()).normal_(generator=generator)

        z = eps.mul(std).add_(mu)
        
        return z

    def encode(self, inp, cond, mask):
        out = self.encoder(inp, cond, mask)
        return out["mu"], out["logvar"]

    def decode(self, inp, cond, mask):
        out = self.decoder(inp, cond, mask)
        return out 

    def textemb(self, emb):
        bs = emb.shape[0]
        text_emb = emb.view(bs,-1) ## [bs, 59136]
        text_emb = self.project(text_emb)
        return text_emb
    
    def forward(self, x, text_emb, mask):
        batch = {}
        batch["x"] = x
        batch["mask"] = mask 
        bs = x.shape[0] # 
       
        text_emb = self.textemb(text_emb)
        batch["y"] = text_emb ## [1, 77, 768]

        ## encode ##
        mu, logvar = self.encode(x, text_emb, mask)
     
        ## batch.update(out) ##
        batch["mu"] = mu
        batch["logvar"] = logvar
        batch["z"] = self.reparameterize(batch) ## [2, 100]
        
        ## decode ##
        out = self.decode(batch["z"], text_emb, mask)
        batch.update(out)
       
        return batch

    def generate_one(self, text_emb):
        z = torch.randn(2*self.latent_dim, device=self.device)[None]
        z = z.reshape((2, self.latent_dim)) # [2, 200]
        text_emb = text_emb.view(text_emb.shape[0],-1)
        text_emb = self.project(text_emb)
        batch = self.decoder(z, text_emb)
        
        return batch["output"]
        
    ### latent ###
    def return_latent_z(self, batch, seed=None):
        return self.encoder(batch)["z"]
    
    def return_latent(self, batch, seed=None):
        distrib_param = self.encoder(batch)
        batch.update(distrib_param)
        return self.reparameterize(batch, seed=seed)
    
    #### loss ####
    def compute_loss(self, batch):
        mixed_loss = 0
        losses = {}
        self.lambdas = {'kl': 1, "rc": 1}
        for ltype, lam in self.lambdas.items():
            loss_function = self.get_loss_function(ltype)
            loss = loss_function(batch)           
            mixed_loss += loss #*lam
            losses[ltype] = loss.item()
        
        mixed_clip_loss, clip_losses = self.compute_clip_losses(batch)
        mixed_loss_with_clip = mixed_loss + mixed_clip_loss  # this is the ultimate loss to optimize, combining ALL losses
        losses["clip"] = clip_losses['text_cosine']
        losses["mixed"] = mixed_loss.item()
       
        return mixed_loss_with_clip, losses
    
    def compute_kl_loss(self, batch):
        mu, logvar = batch["mu"], batch["logvar"]
        loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return loss

    def compute_rc_loss(self, batch):
        mask = batch["mask"]
        gtmasked = batch["x"][mask]
        outmasked = batch["output"][mask] ##[2, 2, 100]
        loss = F.mse_loss(gtmasked, outmasked, reduction='mean')

        return loss
    
    def compute_clip_losses(self, batch):
        mixed_clip_loss = 0.
        clip_losses = {}

        if self.clip_training:
            for d in self.clip_training.split('_'):
                if d == 'image':
                    # features = self.clip_model.encode_image(
                    #     batch['clip_images']).float()  # preprocess is done in dataloader
                    features = None
                    
                elif d == 'text':
                    ## tokenizer ##
                    # texts = clip.tokenize(batch['clip_text']).to(self.device)
                    # features = self.clip_model.encode_text(texts).float()
                    prompt_ids = self.tokenizer(
                            batch['caption'], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).input_ids.to(self.device)
                    features = self.text_encoder(prompt_ids)[0]

                ## normalized features ##
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)

                # logit_scale = self.clip_model.logit_scale.exp()
                logit_scale = self.text_encoder.logit_scale.exp()
                logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                logits_per_d = logits_per_motion.t()

                batch_size = batch['x'].shape[0]
                ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                mixed_clip_loss += clip_mixed_loss

        else:
            for d in self.clip_lambdas.keys():
                if len(self.clip_lambdas[d].keys()) == 0:
                    continue
                with torch.no_grad():
                    if d == 'image':
                        # features = self.clip_model.encode_image(
                        #     batch['clip_images']).float()  # preprocess is done in dataloader
                        features = None 

                    elif d == 'text':
                        # texts = clip.tokenize(batch['clip_text']).to(self.device)
                        # features = self.clip_model.encode_text(texts).float()
                        prompt_ids = self.tokenizer(
                            batch['caption'], max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
                        ).input_ids.to(self.device)
                        features = self.text_encoder(prompt_ids)[0]
                        features = features.view(batch["output"].shape[0],-1) 
                        features = self.project(features)
                    else:
                        raise ValueError(f'Invalid clip domain [{d}]')

                ## normalized features ##
                features_norm = features / features.norm(dim=-1, keepdim=True)
                seq_motion_features_norm = batch["z"] / batch["z"].norm(dim=-1, keepdim=True)

                if 'ce' in self.clip_lambdas[d].keys():
                    # logit_scale = self.clip_model.logit_scale.exp()
                    logit_scale = self.text_encoder.logit_scale.exp()
                    logits_per_motion = logit_scale * seq_motion_features_norm @ features_norm.t()
                    logits_per_d = logits_per_motion.t()

                    batch_size = batch['x'].shape[0]
                    ground_truth = torch.arange(batch_size, dtype=torch.long, device=self.device)

                    ce_from_motion_loss = loss_ce(logits_per_motion, ground_truth)
                    ce_from_d_loss = loss_ce(logits_per_d, ground_truth)
                    clip_mixed_loss = (ce_from_motion_loss + ce_from_d_loss) / 2.

                    clip_losses[f'{d}_ce_from_d'] = ce_from_d_loss.item()
                    clip_losses[f'{d}_ce_from_motion'] = ce_from_motion_loss.item()
                    clip_losses[f'{d}_mixed_ce'] = clip_mixed_loss.item()
                    mixed_clip_loss += clip_mixed_loss * self.clip_lambdas[d]['ce']

                if 'mse' in self.clip_lambdas[d].keys():
                    mse_clip_loss = loss_mse(features, batch["z"])
                    clip_losses[f'{d}_mse'] = mse_clip_loss.item()
                    mixed_clip_loss += mse_clip_loss * self.clip_lambdas[d]['mse']

                if 'cosine' in self.clip_lambdas[d].keys():
                    cos = cosine_sim(features_norm, seq_motion_features_norm)
                    cosine_loss = (1 - cos).mean()
                    clip_losses[f'{d}_cosine'] = cosine_loss.item()
                    mixed_clip_loss += cosine_loss * self.clip_lambdas[d]['cosine']

        return mixed_clip_loss, clip_losses
    

    def get_loss_function(self, ltype):
        return self._matching_[ltype]
    

