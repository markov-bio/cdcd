import einops

import torch
from torch import nn

from numpy import pi



class TimeConditioning(nn.Module):
    def __init__(self, fourier_dim, time_cond_dim):
        super().__init__()

        self.fourier_dim=fourier_dim
        self.time_cond_dim=time_cond_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(fourier_dim, time_cond_dim),
            nn.GELU(),
            nn.Linear(time_cond_dim, time_cond_dim) # because not doing latent token time conditioning
        )


    def forward(self,t:torch.Tensor):
        support_frequencies=torch.linspace(0,pi,self.fourier_dim, device=t.device)
        freqs=self.fourier_dim*einops.einsum(t,support_frequencies, 'b, f -> b f')
        conditioning=torch.sin(freqs)
        return self.time_mlp(conditioning)