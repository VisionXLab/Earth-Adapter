import math
import torch
import torch.nn as nn
from torch import Tensor
from functools import reduce
from operator import mul


class earth_adapter(nn.Module):
    def __init__(
        self,
        dim = 64,
        adapter_layer = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23],
        fft_layer = [],
        with_token = False,
        token_dim = 0,
        cutoff_ratio = 0.3,
        scale = 0.1
    ) -> None:
        super().__init__()
        self.dim = dim
        self.fft_layer = fft_layer
        self.token_dim = token_dim
        self.adapter_layer = adapter_layer
        self.with_token = with_token
        self.cutoff_ratio = cutoff_ratio
        self.scale = nn.Parameter(torch.tensor([scale]*24))
        if self.with_token:
            self.refine_token = nn.Parameter(torch.empty([24, 1024, self.token_dim]))  # layer, token_length, embed_dims
        val = math.sqrt(
            6.0
            / float(
                3 * reduce(mul, (16, 16), 1) + 1024
            )
        )
        self.layer_norm = nn.ModuleList([nn.LayerNorm(1024) for _ in range(24)])

        if self.with_token:
            self.mlp_list1 = nn.ModuleList([nn.Sequential(nn.Linear(1024+self.token_dim, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        else:
            self.mlp_list1 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
        if len(self.fft_layer) > 0:
            self.mlp_list2 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
            self.mlp_list3 = nn.ModuleList([nn.Sequential(nn.Linear(1024, self.dim), nn.ReLU(), nn.Linear(self.dim, 1024)) for _ in range(24)])
            self.router = nn.ModuleList([nn.Linear(1024, 3) for _ in range(24)])
            for mlp in self.mlp_list1+self.mlp_list2+self.mlp_list3:       
                nn.init.kaiming_uniform_(mlp[0].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mlp[2].weight, a=math.sqrt(5))
        else:
            for mlp in self.mlp_list1:       
                nn.init.kaiming_uniform_(mlp[0].weight, a=math.sqrt(5))
                nn.init.kaiming_uniform_(mlp[2].weight, a=math.sqrt(5))
        

    def decompose_fft(self, feats: Tensor, layer: int) -> Tensor:

        feats = feats.permute(1, 2, 0).reshape(feats.shape[1], feats.shape[2], 32,32)

        assert feats.dim() == 4, "(batch_size, channels, height, width)"
        
        fft = torch.fft.fft2(feats, norm='ortho') 
        fft_shifted = torch.fft.fftshift(fft, dim=(-2, -1))  

        batch_size, channels, H, W = fft_shifted.shape
        cutoff = int(min(H, W) * self.cutoff_ratio // 2) 
        mask_low = torch.zeros_like(fft_shifted)
        cx, cy = H // 2, W // 2
        mask_low[:, :, cx - cutoff:cx + cutoff, cy - cutoff:cy + cutoff] = 1  
        mask_high = 1 - mask_low  

        fft_low = fft_shifted * mask_low
        fft_high = fft_shifted * mask_high
        feats_low = torch.fft.ifft2(torch.fft.ifftshift(fft_low, dim=(-2, -1)), norm='ortho').real
        feats_high = torch.fft.ifft2(torch.fft.ifftshift(fft_high, dim=(-2, -1)), norm='ortho').real
        return feats_low, feats_high

    def forward(
        self, feats: Tensor, layer: int, batch_first=False, has_cls_token=True
    ) -> Tensor:
    
        if layer not in self.adapter_layer:
            return feats
        feats = feats.permute(1, 0, 2)
        cls_token, feats = torch.tensor_split(feats, [1], dim=0)  
        if self.with_token:
            tokens = self.refine_token[layer].unsqueeze(1).expand(-1, feats.shape[1], -1)  
            combined_feats = torch.cat([feats, tokens], dim=-1)
        else:
            combined_feats = feats

        if layer not in self.fft_layer:            
            delta_feat = self.mlp_list1[layer](combined_feats)
            feats = feats + self.scale[layer] * delta_feat
        else:
            low_freq,high_freq = self.decompose_fft(feats, layer)
            batch_size, channels, height, width = low_freq.shape
            low_freq = low_freq.reshape(batch_size, channels,-1).permute(2,0,1)
            high_freq = high_freq.reshape(batch_size, channels,-1).permute(2,0,1)
            delta_feat1 = self.mlp_list1[layer](combined_feats)
            delta_feat2 = self.mlp_list2[layer](low_freq)
            delta_feat3 = self.mlp_list3[layer](high_freq)
            router_weights = self.router[layer](feats)  
            router_weights = torch.softmax(router_weights, dim=-1)  
            delta_feat = router_weights[:,:, 0].unsqueeze(-1) * delta_feat1 + \
                         router_weights[:,:, 1].unsqueeze(-1) * delta_feat2 + \
                         router_weights[:,:, 2].unsqueeze(-1) * delta_feat3
            feats = feats + self.scale[layer] * delta_feat
        
        if has_cls_token:
            feats = torch.cat([cls_token, feats], dim=0)
        if batch_first:
            feats = feats.permute(1, 0, 2)
        return feats
