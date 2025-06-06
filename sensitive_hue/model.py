import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from base.metrics import *
from mamba_ssm import Mamba
from sklearn.metrics import roc_auc_score
from .modules.revin import RevIN

class MambaBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.mamba = Mamba(
            d_model=d_model,
            d_state=16,  
            d_conv=2,    
            expand=1     
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        out = self.mamba(x)
        return self.norm(out)


class MambaEncoderLayer(nn.Module):
    def __init__(self, d_model: int, head_num: int, dim_hidden_fc: int, dropout=0.1):
        super(MambaEncoderLayer, self).__init__()
        self.attn = MambaBlock(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_hidden_fc, bias=False),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden_fc, d_model, bias=False),
        )

        self.final_enc = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, src):
        src1 = self.attn(self.norm1(src))
        src = src + self.dropout1(src1)

        src2 = self.ffn(self.norm2(src))
        src = src + self.dropout2(src2)

        src3 = self.final_enc(self.norm3(src))
        src = src + self.dropout3(src3)

        return src


class ResidualDecoder(nn.Module):
    def __init__(self, d_model, f_in, dropout=0.1):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.layer2 = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        self.output_proj = nn.Linear(d_model, f_in)

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x_out = x2 + x 
        x_out = self.output_proj(x_out)
        return x_out


class SensitiveMamba(nn.Module):
    def __init__(self, step_num_in: int, f_in: int, d_model: int, head_num: int, dim_hidden_fc: int,
                 encode_layer_num: int, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Embedding(step_num_in, d_model)        
        self.in_linear = nn.Linear(f_in, d_model)
        self.encoder = nn.ModuleList(
            MambaEncoderLayer(d_model, head_num, dim_hidden_fc, dropout)
            for _ in range(encode_layer_num)
        )
        self.decoder = ResidualDecoder(d_model=d_model, f_in=f_in)
        self.sigma_linear = nn.Linear(d_model, f_in)

        self.rev_in = RevIN(f_in, affine=False)
        self.eps = 1e-5

    def forward(self, trend, return_rep=False):
        trend = self.rev_in(trend, mode='norm')
        h = self.in_linear(trend)

        for encoder in self.encoder:
            h = encoder(h)

        x_recon = self.decoder(h)
        x_recon = self.rev_in(x_recon, mode='denorm')

        log_var_recip = self.sigma_linear(h)

        if return_rep:
            return x_recon, log_var_recip, h 
        else:
            return x_recon, log_var_recip
    
    def __str__(self):
        return SensitiveMamba.__name__


