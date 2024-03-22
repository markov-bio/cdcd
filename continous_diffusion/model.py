import einops
import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math

from .conditioning import TimeConditioning

from .DiT_block import DiTBlock
from .RoPe import RotaryEmbedding
# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class DiffusionTransformer(nn.Module):

    def __init__(self, embed_dim,qkv_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        assert embed_dim%num_heads==0 and embed_dim!=num_heads
        self.rope=RotaryEmbedding(qkv_dim//num_heads)
        self.time_conditioning=TimeConditioning(cond_dim,cond_dim)

        self.DiT_blocks=nn.Sequential(*[DiTBlock(embed_dim,qkv_dim,num_heads,cond_dim,self.rope,max_len) for _ in range(n_blocks)])

    def forward(self, x: Tensor, sigma:Tensor,attn_mask:Tensor=None) -> Tensor:
        conditioning=self.time_conditioning(sigma)
        attn_mask= transform_attn_mask(attn_mask)
        x = F.normalize(x,p=2,dim=-1)#critical modification
        for block in self.DiT_blocks:
            x=x+block(x, conditioning, attn_mask)
        return x
    
    
def transform_attn_mask(attn_mask):
    return einops.einsum(attn_mask,attn_mask,'b l, b m -> b l m').unsqueeze(1)