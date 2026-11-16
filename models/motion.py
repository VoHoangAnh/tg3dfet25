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

gpu_id = 0
device = f'cuda:{gpu_id}'
device = torch.device(device if torch.cuda.is_available() else 'cpu')
LATENT_DIM = 16 
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
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim):
        super(EXP_CVAE, self).__init__()
        self.n_d = n_exp*n_dim
        self.latent_dim = latent_dim
        self.n_exp = n_exp 
        self.exp_dim = n_dim
        self.project = nn.Linear(59136, self.n_d)
        self.encoder = EXP_Encoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)
        self.decoder = EXP_Decoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)

    def emb(self, cond):
        cond = cond.view(cond.shape[0],-1) ## [1, 59136] = [1, 77, 768]
        cond = F.relu(self.project(cond)) ## [1, 100]
        return cond 

    def sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(LATENT_DIM, device=device), torch.ones(LATENT_DIM, device=device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    

        return t

    def mloss(self, X, X_hat, mean, logvar, state = "exp"): 
        X = X.squeeze(1)
        
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='mean')
        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.pow(2))
        total_loss = reconstruction_loss + KL_divergence

        return total_loss, reconstruction_loss, KL_divergence

    def encode(self, inp, cond):
        cond = self.emb(cond)
        t_mean, t_log_var = self.encoder(inp, cond)
        return t_mean, t_log_var 
    
    def decode(self, z, cond):
        out = self.decoder(z, cond)
        return out 
    
    def forward(self, inp, cond):
        cond = self.emb(cond)
        ## encoder 
        t_mean, t_log_var = self.encoder(inp, cond)
        ## sampling  
        z = self.sampling(t_mean, t_log_var)
        ## decoder 
        output = self.decoder(z, cond)
        return output, t_mean, t_log_var 

class POSE_CVAE(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, n_dim):
        super(POSE_CVAE, self).__init__()
        self.n_d = n_exp*n_dim
        self.latent_dim = latent_dim
        self.n_exp = n_exp 
        self.exp_dim = n_dim
        
        self.project = nn.Linear(59136, self.n_d)
        self.encoder = POSE_Encoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)
        self.decoder = POSE_Decoder(intermediate_dim, latent_dim, n_exp, n_dim).to(device)

    def emb(self, cond):
        cond = cond.view(cond.shape[0],-1)
        cond = F.relu(self.project(cond)) ## [1, 100]
        return cond 

    def sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(LATENT_DIM, device=device), torch.ones(LATENT_DIM, device=device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    

        return t

    def mloss(self, X, X_hat, mean, logvar, state="exp"): 
        X = X.squeeze(1)
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='mean')
        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.pow(2))
        total_loss = reconstruction_loss + KL_divergence

        return total_loss, reconstruction_loss, KL_divergence

    def encode(self, inp, cond):
        cond = self.emb(cond)
        t_mean, t_log_var = self.encoder(inp, cond)
        return t_mean, t_log_var 
    
    def decode(self, z, cond):
        out = self.decoder(z, cond)
        return out 
    
    def forward(self, inp, cond):
        cond = self.emb(cond)
        ## encoder 
        t_mean, t_log_var = self.encoder(inp, cond)
        ## sampling  
        z = self.sampling(t_mean, t_log_var)
        ## decoder 
        output = self.decoder(z, cond)
        return output, t_mean, t_log_var 

###########################################################    
#### AutoEncoder (FC) - NEUTRAL ###
class NEUTRAL_Encoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, exp_dim, pose_dim, cam_dim): ## 100, 256 
        super(NEUTRAL_Encoder, self).__init__()
        self.n_d = n_exp*(exp_dim + pose_dim) 
        self.latent_dim = latent_dim 

        self.out1 = nn.Linear(self.n_d, intermediate_dim) ## [200, 128] 
        self.out2 = nn.Linear(intermediate_dim, intermediate_dim//2) ## [128, 64]
        self.out3 = nn.Linear(intermediate_dim//2, intermediate_dim//4) ## [128, 64]
        self.out4 = nn.Linear(intermediate_dim//4, 2 * latent_dim) ## [128, 64]
       

    def forward(self, inp):
        ## expression ##  
        output = F.gelu(self.out1(inp))
        output = F.gelu(self.out2(output))
        output = F.gelu(self.out3(output))
        output = self.out4(output)

        t_mean = output[:,0:self.latent_dim]
        t_log_var = output[:,self.latent_dim:] 
        
        return  t_mean, t_log_var 
    
class NEUTRAL_Decoder(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, exp_dim, pose_dim, cam_dim):
        super(NEUTRAL_Decoder, self).__init__()
        n_d1 = n_exp*exp_dim
        n_d2 = n_exp*pose_dim 
        n_d3 = n_exp*cam_dim 

        self.latent_dim = latent_dim
        
        ## Expression ##  
        self.out1_exp = nn.Linear(latent_dim, intermediate_dim//4) ## [200, 128]
        self.out2_exp = nn.Linear(intermediate_dim//4, intermediate_dim//2) ## [64, 128]
        self.out3_exp = nn.Linear(intermediate_dim//2, intermediate_dim) ## [64, 128]
        self.out4_exp = nn.Linear(intermediate_dim, n_d1) ## [128, 200]

        ## Pose ##
        self.out1_pose = nn.Linear(latent_dim, intermediate_dim//4) ## [200, 128]
        self.out2_pose = nn.Linear(intermediate_dim//4, intermediate_dim//2) ## [64, 128]
        self.out3_pose = nn.Linear(intermediate_dim//2, intermediate_dim) ## [64, 128]
        self.out4_pose = nn.Linear(intermediate_dim, n_d2) ## [128, 200]

    def forward(self, inp):        
        ## Expression ##
        output_exp = F.gelu(self.out1_exp(inp))
        output_exp = F.gelu(self.out2_exp(output_exp))
        output_exp = F.gelu(self.out3_exp(output_exp))
        output_exp = self.out4_exp(output_exp)   

        ## Pose ##
        output_pose = F.gelu(self.out1_pose(inp))
        output_pose = F.gelu(self.out2_pose(output_pose))
        output_pose = F.gelu(self.out3_pose(output_pose)) 
        output_pose = self.out4_pose(output_pose)  

        return output_exp, output_pose #, output_cam  


class NEUTRAL_CVAE(nn.Module):
    def __init__(self, intermediate_dim, latent_dim, n_exp, exp_dim, pose_dim, cam_dim):
        super(NEUTRAL_CVAE, self).__init__()
        self.n_d = n_exp*(exp_dim + pose_dim)
        self.latent_dim = latent_dim
        self.n_exp = n_exp 
        
        self.encoder = NEUTRAL_Encoder(intermediate_dim, latent_dim, n_exp, exp_dim, pose_dim, cam_dim).to(device)
        self.decoder = NEUTRAL_Decoder(intermediate_dim, latent_dim, n_exp, exp_dim, pose_dim, cam_dim).to(device)

    def sampling(self, t_mean, t_log_var):
        ep = torch.normal(torch.zeros(LATENT_DIM, device=device), torch.ones(LATENT_DIM, device=device)) ## [16]
        t = (ep * torch.exp(t_log_var/2)) + t_mean    

        return t

    def mloss(self, X, X_hat, mean, logvar, state="exp"): 
        reconstruction_loss = F.mse_loss(X, X_hat, reduction='mean')
        KL_divergence = -0.5 * torch.sum(1 + logvar - logvar.exp() - mean.pow(2))
        total_loss = reconstruction_loss + KL_divergence

        return total_loss, reconstruction_loss, KL_divergence

    def encode(self, inp):
        t_mean, t_log_var  = self.encoder(inp)
        return t_mean, t_log_var
    
    def decode(self, z):
        out_exp, out_pose = self.decoder(z)
        return out_exp, out_pose 
    
    def forward(self, inp):
        ## encoder ## 
        t_mean, t_log_var = self.encoder(inp)
        ## sampling ##  
        z = self.sampling(t_mean, t_log_var)
        ## decoder ##
        output_exp, output_pose = self.decoder(z)
        
        return output_exp, output_pose, t_mean, t_log_var
    

