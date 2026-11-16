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
from einops import rearrange, repeat
from torch import nn, einsum

from decalib.deca import DECA
from decalib.utils.config import cfg as deca_cfg
from decalib.utils.util import batch_orth_proj, vertex_normals


CLIP_IMG_DIM = 1024
class TemporalConvNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        layers = []
        kernel_size = cfg.kernel_size
        
        for i in range(self.cfg.num_layers):
            layers += [torch.nn.Conv1d(self.cfg.feature_dim, 
                    self.cfg.feature_dim, kernel_size=kernel_size, padding='same',)]
            layers += [torch.nn.ReLU()]
            if self.cfg.dropout > 0:
                layers += [torch.nn.Dropout(p=self.cfg.dropout)] 
            if self.cfg.norm_layer: 
                if self.cfg.norm_layer == "batchnorm1d":
                    layers += [torch.nn.BatchNorm1d(self.cfg.feature_dim)]
                elif self.cfg.norm_layer == "layernorm":
                    layers += [torch.nn.LayerNorm(self.cfg.feature_dim)]
                elif self.cfg.norm_layer == "instancenorm1d":
                    layers += [torch.nn.InstanceNorm1d(self.cfg.feature_dim)]
                else:
                    raise NotImplementedError("Unknown norm layer: {}".format(self.cfg.norm_layer))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, sample):
        feat = sample["fused_feature"].transpose(1, 2)
        out = self.layers(feat).transpose(1, 2) 
        sample["seq_encoder_output"] = out 
        return sample 
        
    def get_trainable_parameters(self): 
        return list(self.parameters())

    def input_feature_dim(self):
        return self.cfg.feature_dim

    def output_feature_dim(self):
        return self.cfg.feature_dim 

#############################################################    
######              TEXT MAPPING                        #####
BN_MOMENTUM = 0.1
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)
    
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., selfatt=True):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, z=None):
        qkv = self.to_qkv(x).chunk(3, dim=-1)  # (B, N, inner_dim)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale   # (B, Nq, Nk)
        attn = self.attend(dots)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)

class CrossAttention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_k = nn.Linear(dim, inner_dim , bias=False)
        self.to_v = nn.Linear(dim, inner_dim , bias = False)
        self.to_q = nn.Linear(dim, inner_dim, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x_qkv):
        b, n, _, h = *x_qkv.shape, self.heads

        ### Key and Values from sequence S1
        k = self.to_k(x_qkv)
        k = rearrange(k, 'b n (h d) -> b h n d', h = h) ## sequence S1

        v = self.to_v(x_qkv)
        v = rearrange(v, 'b n (h d) -> b h n d', h = h) ## sequence S1

        ### Query from sequence S2
        q = self.to_q(x_qkv[:, 0].unsqueeze(1)) ## sequence S2
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        ### attention matrix: query and key  
        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale
        attn = dots.softmax(dim=-1)
        
        #### apply values to attention matrix ###############
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out =  self.to_out(out)

        return out
    
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0., selfatt=True):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head,
                                       dropout=dropout, selfatt=selfatt)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, z=None):
        for attn, ff in self.layers:
            x = attn(x, z=z) + x
            x = ff(x) + x
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, par_dim = 96, par_depth = 4, par_heads =3, par_dim_head = 32, par_mlp_dim = 384,
                    text_dim = 192, text_depth = 1, text_heads = 3, text_dim_head = 64, text_mlp_dim = 768,
                    cross_attn_depth = 1, cross_attn_heads = 3, dropout = 0.):
            super().__init__()
            
            self.transformer_enc_par = Transformer(par_dim, par_depth, par_heads, par_dim_head, par_mlp_dim)
            self.transformer_enc_text = Transformer(text_dim, text_depth, text_heads, text_dim_head, text_mlp_dim)
            self.mapping = nn.Linear(59136, 768)
            self.cross_attn_layers = nn.ModuleList([])
            for _ in range(cross_attn_depth):
                self.cross_attn_layers.append(nn.ModuleList([
                    nn.Linear(par_dim, text_dim),
                    nn.Linear(text_dim, par_dim),
                    PreNorm(text_dim, CrossAttention(text_dim, heads = cross_attn_heads, dim_head = text_dim_head, dropout = dropout)),
                    nn.Linear(text_dim, par_dim),
                    nn.Linear(par_dim, text_dim),
                    PreNorm(par_dim, CrossAttention(par_dim, heads = cross_attn_heads, dim_head = par_dim_head, dropout = dropout)),
                ]))


    def forward(self, x_par, x_text):
        x_par = self.transformer_enc_par(x_par) ## [bs, 2, 53]
        x_text = x_text.view(x_text.shape[0], -1)
        x_text = x_text.unsqueeze(1)
        x_text = torch.cat((x_text, x_text), 1) ## [bs, 2, 59136]
        x_text = self.mapping(x_text)
        
        x_text = self.transformer_enc_text(x_text) ## [bs, 2, 768]

        for f_par2text, g_text2par, cross_attn_par, f_text2par, g_par2text, cross_attn_text in self.cross_attn_layers:
            par_cls = x_par[:, 0]
            x_par = x_par[:, 1:]
            text_cls = x_text[:, 0]
            x_text = x_text[:, 1:]

            # Cross Attn for parameter #
            cal_q = f_text2par(text_cls.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_par), dim=1)
            cal_out = cal_q + cross_attn_text(cal_qkv)
            cal_out = g_par2text(cal_out)
            x_text = torch.cat((cal_out, x_text), dim=1)

            # Cross Attn for text #
            cal_q = f_par2text(par_cls.unsqueeze(1))
            cal_qkv = torch.cat((cal_q, x_text), dim=1)
            cal_out = cal_q + cross_attn_par(cal_qkv)
            cal_out = g_text2par(cal_out)
            x_par = torch.cat((cal_out, x_par), dim=1)

        return x_par, x_text
    
class TFED(nn.Module):
    def __init__(
        self,
        device,
        num_frames=None,
        in_channels=768,
        out_channels=768,
        out_channels_par = 53,
        n_heads = 4,
        num_layers=1, # 2
        cross_attention_dim=768,
    ):
        super().__init__()
        MAX_LENGTH  = 1024
        d_head = out_channels//n_heads
        self.trf_blocks = nn.ModuleList([])
        self.num_frames = num_frames
        out_channels_1 = out_channels
        out_channels_2 = out_channels_par
        EMB_DIM = 128
        self.exp_dim = 50 
        self.pose_dim = 6
        self.latent_dim = 16
        self.device = device 
        self.exp_map = nn.Linear(self.latent_dim, self.num_frames*self.exp_dim).to(device)
        self.pose_map = nn.Linear(self.latent_dim, self.num_frames*self.pose_dim).to(device)

        for n in range(num_layers):
            self.trf_blocks.append(TransformerEncoder(
                    ### Modifing par_depth, text_depth
                    par_dim = 53, par_depth = 2, par_heads =3, par_dim_head = 32, par_mlp_dim = 53,
                    text_dim = 768, text_depth = 1, text_heads = 3, text_dim_head = 32, text_mlp_dim = 768))
          

        self.norm1 = nn.LayerNorm(out_channels_1)
        self.norm2 = nn.LayerNorm(out_channels_2)

        self.mapping_pose = nn.Linear(768+53, 6)
        self.mapping_exp = nn.Linear(768+53, 50)

    def set_numframe(self,num_frames):
        self.num_frames = num_frames

    def exp_sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(self.latent_dim, device=self.device), torch.ones(self.latent_dim, device=self.device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    
        t = self.exp_map(t)

        return t
    
    def pose_sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(self.latent_dim, device=self.device), torch.ones(self.latent_dim, device=self.device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    
        t = self.pose_map(t)
        return t
    
    def feat(self, in_exp, in_pose, context):
        b,l,c = context.shape ## [bs, 77, 768]
        
        in_exp = in_exp[:b,:].reshape((b, self.num_frames, in_exp.shape[2]//self.num_frames)) ## [bs, n_frames, 50]
        in_pose = in_pose[:b,:].reshape((b, self.num_frames, in_pose.shape[2]//self.num_frames)) ## [bs, n_frames, 6]
       
        in_params = torch.cat((in_exp, in_pose[:b,:,:3]), 2) ## [bs, n_frames, 53]
        x_par = in_params
        
        for trf_blk in self.trf_blocks:
            x_par, x_text = trf_blk(in_params, context)

        x_text = self.norm1(x_text)
        x_par = self.norm2(x_par)
        x = torch.cat((x_text, x_par),2)

        return x 
    
    def mloss(self, exp_emb, pose_emb, fake_exp_emb, fake_pose_emb):
        rec_exp_loss = F.mse_loss(exp_emb, fake_exp_emb, reduction='mean')
        rec_pose_loss = F.mse_loss(pose_emb, fake_pose_emb, reduction='mean')
        return rec_exp_loss, rec_pose_loss 

    def forward(self, in_exp, in_pose, context):
        b,l,c = context.shape ## [bs, 77, 768]
        x = self.feat(in_exp, in_pose, context)
        pose_emb = self.mapping_pose(x) ## [bs, 2, 6]
        exp_emb = self.mapping_exp(x) ## [bs, 2, 50]

        return exp_emb, pose_emb

#############################################################    
###### Conditional Variational AutoEncoder (FC) - EXP #######
class EXP_Encoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, exp_dim): ## 100, 256 
        super(EXP_Encoder, self).__init__()
        self.n_d = n_exp*exp_dim 
        self.latent_dim = latent_dim 
        self.out1 = nn.Linear(2*self.n_d, intermediate_dim) ## [200, 128] 
        self.out2 = nn.Linear(intermediate_dim, intermediate_dim//2) ## [128, 64]
        self.out3 = nn.Linear(intermediate_dim//2, 2 * latent_dim) ## [64, 2 x 16]

    def forward(self, inp_exp, cond):
        ## expression ##  
        inp_exp = inp_exp.squeeze(1)
        inp = torch.cat([inp_exp, cond], dim=1) ## [1, 200]

        output = F.gelu(self.out1(inp))
        output = F.gelu(self.out2(output))
        output = self.out3(F.gelu(output))
        t_mean = output[:,0:self.latent_dim]
        t_log_var = output[:,self.latent_dim:] 
        
        return  t_mean, t_log_var


class EXP_Decoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, exp_dim):
        super(EXP_Decoder, self).__init__()
        n_d1 = n_exp*exp_dim
        self.latent_dim = latent_dim
        
        ## Expression ##  
        self.out1_exp = nn.Linear(n_d1 + latent_dim, intermediate_dim//2) ## [200, 128]
        self.out2_exp = nn.Linear(intermediate_dim//2, intermediate_dim) ## [64, 128]
        self.out3_exp = nn.Linear(intermediate_dim, n_exp*exp_dim) ## [128, 200]

    def forward(self, inp, cond):
        inp = torch.cat([inp, cond], dim=1) ## [noise + cond]        
        ## Expression ##
        output_exp = F.gelu(self.out1_exp(inp))
        output_exp = F.gelu(self.out2_exp(output_exp))
        output_exp = self.out3_exp(output_exp)        
        
        return output_exp 

###### Conditional Variational AutoEncoder (FC) - POSE ########
class POSE_Encoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim): ## 9, 256 
        super(POSE_Encoder, self).__init__()
        self.n_d = n_exp*n_dim 
        self.latent_dim = latent_dim 
        self.out1 = nn.Linear(2*self.n_d, intermediate_dim) ## [18, 128] 
        self.out2 = nn.Linear(intermediate_dim, intermediate_dim//2) ## [128, 64]
        self.out3 = nn.Linear(intermediate_dim//2, 2 * latent_dim) ## [64, 2 x 16]

    def forward(self, inp, cond):
        ## Expression ##  
        inp = inp.squeeze(1)
        inp = torch.cat([inp, cond], dim=1) ## [1, 200]
        output = F.gelu(self.out1(inp))
        output = F.gelu(self.out2(output))
        output = self.out3(F.gelu(output))
        t_mean = output[:,0:self.latent_dim]
        t_log_var = output[:,self.latent_dim:] 
        
        return  t_mean, t_log_var

class POSE_Decoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim):
        super(POSE_Decoder, self).__init__()
        n_d1 = n_exp*n_dim
        self.latent_dim = latent_dim
        ## Pose ##  
        self.out1_pose = nn.Linear(n_d1 + latent_dim, intermediate_dim//2) ## [200, 128]
        self.out2_pose = nn.Linear(intermediate_dim//2, intermediate_dim) ## [64, 128]
        self.out3_pose = nn.Linear(intermediate_dim, n_exp*n_dim) ## [128, 200]

    def forward(self, inp, cond):
        inp = torch.cat([inp, cond], dim=1) ## [noise + cond]        
        ## Pose ##
        output_exp = F.gelu(self.out1_pose(inp))
        output_exp = F.gelu(self.out2_pose(output_exp))
        output_exp = self.out3_pose(output_exp)        
        
        return output_exp 

class EXP_CVAE(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim, device):
        super(EXP_CVAE, self).__init__()
        self.n_d = n_exp*n_dim
        self.latent_dim = latent_dim
        self.device = device 
        self.n_exp = n_exp 
        self.exp_dim = n_dim
        
        self.encoder = EXP_Encoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)
        self.decoder = EXP_Decoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)

    def emb(self, cond):
        cond = cond.view(cond.shape[0],-1) ## [1, 59136] = [1, 77, 768]
        cond = F.relu(self.project(cond)) ## [1, 100]
        return cond 

    def sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(self.latent_dim, device=self.device), torch.ones(self.latent_dim, device=self.device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    
        return t
    
    def mloss(self, X, X_hat, mean, logvar): 
        X = X.squeeze(1)
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='mean')
        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.pow(2))
        total_loss = reconstruction_loss + KL_divergence

        return total_loss, reconstruction_loss, KL_divergence

    def encode(self, inp, cond):
        t_mean, t_log_var = self.encoder(inp, cond)
        return t_mean, t_log_var 
    
    def decode(self, z, cond):
        out = self.decoder(z, cond)
        return out 
    
    def forward(self, inp, cond):
        ## encoder 
        t_mean, t_log_var = self.encoder(inp, cond)
        ## sampling  
        z = self.sampling(t_mean, t_log_var)
        ## decoder 
        output = self.decoder(z, cond)
        return output, t_mean, t_log_var 


class POSE_CVAE(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim, device):
        super(POSE_CVAE, self).__init__()
        self.n_d = n_exp*n_dim
        self.latent_dim = latent_dim
        self.device = device 
        self.n_exp = n_exp 
        self.pose_dim = n_dim
        
        self.encoder = POSE_Encoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)
        self.decoder = POSE_Decoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)

    def emb(self, cond):
        cond = cond.view(cond.shape[0],-1)
        cond = F.relu(self.project(cond)) ## [1, 100]
        return cond 

    def sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(self.latent_dim, device=self.device), torch.ones(self.latent_dim, device=self.device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    

        return t
    
    def mloss(self, X, X_hat, mean, logvar, state="exp"): 
        X = X.squeeze(1)
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='mean')
        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.pow(2))
        total_loss = reconstruction_loss + KL_divergence

        return total_loss, reconstruction_loss, KL_divergence

    def encode(self, inp, cond):
        t_mean, t_log_var = self.encoder(inp, cond)
        return t_mean, t_log_var 
    
    def decode(self, z, cond):
        out = self.decoder(z, cond)
        return out 
    
    def forward(self, inp, cond):
        ## encoder 
        t_mean, t_log_var = self.encoder(inp, cond)
        ## sampling  
        z = self.sampling(t_mean, t_log_var)
        ## decoder 
        output = self.decoder(z, cond)

        return output, t_mean, t_log_var 

class FET_MODEL(nn.Module):
    def __init__(self, inter_dim, latent_dim, n_exp, device, deca_finetune=False):
        super(FET_MODEL, self).__init__()
        self.latent_dim = latent_dim
        self.inter_dim = inter_dim 
        self.n_exp = n_exp
        self.device = device 
        self.cvae_exp = EXP_CVAE(self.inter_dim, self.latent_dim, n_exp=2, n_dim=50, device = device).to(device)
        self.cvae_pose = POSE_CVAE(self.inter_dim, self.latent_dim, n_exp=2, n_dim=6, device = device).to(device)
        self.fa_text_deform = TFED(device, num_frames = n_exp).to(device)
       
        # >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> DECA model >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> #
        print("\n\n ************ Load pre-traiend DECA ************")
        self.deca = DECA(config=deca_cfg)
        self.deca.eval()
        self.deca_size = [224, 224]  
        if not deca_finetune:
            for param in self.deca.parameters():
                param.requires_grad = False

    ## loss ##
    def mloss(self, pred_verts, src_verts):
        X0 = src_verts[0]['verts']
        X_hat0 = pred_verts[0]['verts']
        X0 = X0.squeeze(1)
    
        X1 = src_verts[1]['verts']
        X_hat1 = pred_verts[1]['verts']
        X1 = X1.squeeze(1)

        rec_loss = F.mse_loss(X0, X_hat0, reduction='mean') + F.mse_loss(X1, X_hat1, reduction='mean') 

        return rec_loss 

    def decode(self, cond):
        n_samples = cond.shape[0]
        exp_sample = np.random.normal(0, 1, size=(n_samples, self.latent_dim)) ## [b, 16]
        pose_sample = np.random.normal(0, 1, size=(n_samples, self.latent_dim)) ## [b, 16]
        z_exp = torch.tensor(exp_sample, dtype=torch.float, device=self.device)
        z_pose = torch.tensor(pose_sample, dtype=torch.float, device=self.device)
        
        z_fake_exp = self.fa_text_deform.exp_map(z_exp)
        z_fake_pose = self.fa_text_deform.pose_map(z_pose)
        z_fake_exp = z_fake_exp.unsqueeze(1)
        z_fake_pose = z_fake_pose.unsqueeze(1)
       
        fake_exp_emb, fake_pose_emb = self.fa_text_deform(z_fake_exp, z_fake_pose, cond)
        fake_exp_emb = fake_exp_emb.view(fake_exp_emb.shape[0], -1) ## [bs, n_frames*50]
        fake_pose_emb = fake_pose_emb.view(fake_pose_emb.shape[0], -1) ## [bs, n_frames*6]
        out_exp = self.cvae_exp.decode(z_exp, fake_exp_emb)
        out_pose = self.cvae_pose.decode(z_pose, fake_pose_emb)

        return out_exp, out_pose 
    
    def encode_verts(self, out_exp, out_pose, in_exp, in_pose, in_shape, in_cam):
        out_pose = out_pose.reshape((out_pose.shape[0], self.n_exp, 6))
        out_exp = out_exp.reshape((out_exp.shape[0], self.n_exp, 50))
        in_pose = in_pose.reshape((in_pose.shape[0], self.n_exp, 6))
        in_exp = in_exp.reshape((in_exp.shape[0], self.n_exp, 50))
        in_shape = in_shape.reshape((in_shape.shape[0], self.n_exp, 100))
        in_cam = in_cam.reshape((in_cam.shape[0], self.n_exp, 3))
        
        pred_verts = []
        src_verts = []
        
        for i in range(self.n_exp):
            pred_code = {}
            pred_code['pose'] = out_pose[:,i,:].squeeze(1) 
            pred_code['exp'] = out_exp[:,i,:].squeeze(1) 
            pred_code['shape'] = in_shape[:,i,:].squeeze(1) 
            pred_code['cam'] = in_cam[:,i,:].squeeze(1) 
            i_pred_verts = self.deca.decode_vert(pred_code, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            src_code = {}
            src_code['pose'] = in_pose[:,i,:].squeeze(1)  
            src_code['exp'] = in_exp[:,i,:].squeeze(1) 
            src_code['shape'] = in_shape[:,i,:].squeeze(1) 
            src_code['cam'] = in_cam[:,i,:].squeeze(1) 
            i_src_verts = self.deca.decode_vert(src_code, rendering=False, vis_lmk=False, return_vis=False, use_detail=False)
            
            pred_verts.append(i_pred_verts)
            src_verts.append(i_src_verts)
        
        return pred_verts, src_verts 
    
    ## ind: parameters (exp, pose) - cond: text ##
    def forward(self, in_exp, in_pose, in_shape, in_cam, cond):
        ### TFED ### 
        exp_emb, pose_emb = self.fa_text_deform(in_exp, in_pose, cond) ## [bs, n_frames, 50/6]
        exp_emb = exp_emb.view(exp_emb.shape[0], -1) ## [bs, n_frames*50]
        pose_emb = pose_emb.view(pose_emb.shape[0], -1) ## [bs, n_frames*6]

        ### Expression & Pose encoders ###
        exp_mean, exp_log_var = self.cvae_exp.encode(in_exp, exp_emb)
        pose_mean, pose_log_var = self.cvae_pose.encode(in_pose, pose_emb)
        
        ### Expression & Pose decoders ###
        z_exp_feat = self.fa_text_deform.exp_sampling(exp_mean, exp_log_var)
        z_pose_feat = self.fa_text_deform.pose_sampling(exp_mean, exp_log_var)
        z_exp_feat = z_exp_feat.unsqueeze(1)
        z_pose_feat = z_pose_feat.unsqueeze(1)
        z_exp = self.cvae_exp.sampling(exp_mean, exp_log_var) ## [bs, 16]
        z_pose = self.cvae_pose.sampling(exp_mean, exp_log_var)
        
        fake_exp_emb, fake_pose_emb = self.fa_text_deform(z_exp_feat, z_pose_feat, cond) ## [bs, n_frames, 50/6]
        fake_exp_emb = fake_exp_emb.view(fake_exp_emb.shape[0], -1) ## [bs, n_frames*50]
        fake_pose_emb = fake_pose_emb.view(fake_pose_emb.shape[0], -1) ## [bs, n_frames*6]
        out_exp = self.cvae_exp.decode(z_exp, fake_exp_emb)
        out_pose = self.cvae_pose.decode(z_pose, fake_pose_emb)

        ### Verts ###
        pred_verts, src_verts = self.encode_verts(out_exp, out_pose, in_exp, in_pose, in_shape, in_cam)

        return out_exp, exp_mean, exp_log_var, out_pose, pose_mean, pose_log_var, \
            fake_exp_emb, fake_pose_emb, exp_emb, pose_emb, pred_verts, src_verts  


def compute_pitch_yaw(R):
    x, y, z = R[..., 2, 0], R[..., 2, 1], R[..., 2, 2]
    yaw = torch.atan2(x, z) * 180 / np.pi
    pitch = torch.atan2(y, z) * 180 / np.pi
    return pitch[:, None], yaw[:, None]


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    ''' Calculates the rotation matrices for a batch of rotation vectors
        Parameters
        ----------
        rot_vecs: torch.tensor Nx3
            array of N axis-angle vectors
        Returns
        -------
        R: torch.tensor Nx3x3
            The rotation matrices for the given axis-angle parameters
    '''

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + 1e-8, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)

    return rot_mat

