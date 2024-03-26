import math
import torch
from torch import nn
from torch.nn import functional as F
import einops

from .RoPe import RotaryEmbedding

class SelfAttention(nn.Module):
    def __init__(self,embed_dim:int,qkv_dim:int,num_heads:int,rope:RotaryEmbedding):
        super().__init__()

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.linear=nn.Linear(embed_dim,3*qkv_dim) #this can be generalized

        self.feedforward=nn.Linear(qkv_dim,embed_dim)
        nn.init.zeros_(self.feedforward.weight)
        nn.init.zeros_(self.feedforward.bias)
        
        self.rope=rope

        self.scale=nn.Parameter(torch.tensor(0.))

    def forward(self,x,attn_mask=None):

        x=self.linear(x)
        x=einops.rearrange(x,'... l (h c) -> ... h l c', h=self.num_heads)       
        q,k,v=x.chunk(3,dim=-1)

        # Using QK-norm
        q=F.normalize(q,p=2,dim=-1)*torch.exp(self.scale)
        k=F.normalize(k,p=2,dim=-1)

        q,k=self.rope(q,k)

        x=F.scaled_dot_product_attention(q,k,v, scale=1, attn_mask=attn_mask)

        x=einops.rearrange(x,'... h l c -> ... l (h c)')

        return self.feedforward(x)