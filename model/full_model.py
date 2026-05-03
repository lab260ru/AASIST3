#!/usr/bin/env python

import torch
import torch.nn.functional as F
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from .kan import KANLinear
from .gat import GraphAttentionLayer
from .pool import GraphPool
from .branch import InferenceBranch
from .residual import Residual_block
from .wav2vec import Wav2Vec2Encoder

import warnings

def warn_deprecated_model():
    msg = (
        "\n"
        "⚠️  Deprecation Notice\n"
        "   This model is outdated and no longer maintained.\n"
        "   ➤ Please use the updated version: lab260/Spectra-AASIST3\n"
        "   🔗 https://huggingface.co/lab260/Spectra-AASIST3\n"
    )
    warnings.warn(msg, category=DeprecationWarning, stacklevel=2)


class aasist3(nn.Module, PyTorchModelHubMixin):
    def __init__(self, d_args={
        "architecture": "AASIST",
        "nb_samp": 64600,
        "first_conv": 128,
        "filts": [70, [1, 32], [32, 32], [32, 64], [64, 64]],
        "gat_dims": [64, 32],
        "pool_ratios": [0.5, 0.7, 0.5, 0.5],
        "temperatures": [2.0, 2.0, 100.0, 100.0],
    }, size=200, w2v_cache_dir="weights/", load_pretrained=True):
        super().__init__()
        warn_deprecated_model()
        self.w2v_encoder = Wav2Vec2Encoder(cache_dir=w2v_cache_dir, load_pretrained=load_pretrained)
        self.bridge = KANLinear(1024, 128)

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)

        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])))

        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.pos_T = nn.Parameter(torch.randn(1, 67, filts[-1][-1]))

        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0], size=size)
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1], size=size)

        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3, size=size)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3, size=size)

        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master3 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master4 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.inference_branch1 = InferenceBranch(
            gat_dims=gat_dims,
            temperature=temperatures[2],
            pool_ratio=pool_ratios[2],
            size=size
        )
        self.inference_branch2 = InferenceBranch(
            gat_dims=gat_dims,
            temperature=temperatures[2],
            pool_ratio=pool_ratios[2],
            size=size
        )
        self.inference_branch3 = InferenceBranch(
            gat_dims=gat_dims,
            temperature=temperatures[2],
            pool_ratio=pool_ratios[2],
            size=size
        )
        self.inference_branch4 = InferenceBranch(
            gat_dims=gat_dims,
            temperature=temperatures[2],
            pool_ratio=pool_ratios[2],
            size=size
        )

        self.out_layer = KANLinear(5 * gat_dims[1], 2)

    def forward(self, x, Freq_aug=False):
        x = self.w2v_encoder(x)
        x = self.bridge(x)
        x = x.transpose(1, 2)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        e = self.encoder(x)

        # GAT-S
        e_S, _ = torch.max(torch.abs(e), dim=3)
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # GAT-T
        e_T, _ = torch.max(torch.abs(e), dim=2)
        e_T = e_T.transpose(1, 2) + self.pos_T
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        out_T1, out_S1, master1 = self.inference_branch1(out_T, out_S, self.master1)
        out_T2, out_S2, master2 = self.inference_branch2(out_T, out_S, self.master2)
        out_T3, out_S3, master3 = self.inference_branch3(out_T, out_S, self.master3)
        out_T4, out_S4, master4 = self.inference_branch4(out_T, out_S, self.master4)

        out_T1, out_T2 = self.drop_way(out_T1), self.drop_way(out_T2)
        out_T3, out_T4 = self.drop_way(out_T3), self.drop_way(out_T4)
        out_S1, out_S2 = self.drop_way(out_S1), self.drop_way(out_S2)
        out_S3, out_S4 = self.drop_way(out_S3), self.drop_way(out_S4)
        master1, master2 = self.drop_way(master1), self.drop_way(master2)
        master3, master4 = self.drop_way(master3), self.drop_way(master4)

        out_T = torch.stack([out_T1, out_T2, out_T3, out_T4]).max(dim=0)[0]
        out_S = torch.stack([out_S1, out_S2, out_S3, out_S4]).max(dim=0)[0]
        master = torch.stack([master1, master2, master3, master4]).max(dim=0)[0]

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return output
