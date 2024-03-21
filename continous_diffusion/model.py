import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math

from .DiT_block import DiTBlock
from .RoPe import RotaryEmbedding
# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):

    def __init__(self, embed_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        assert embed_dim%num_heads==0 and embed_dim!=num_heads
        self.rope=RotaryEmbedding(embed_dim//num_heads)

        self.DiT_blocks=nn.Sequential(*[DiTBlock(embed_dim,num_heads,cond_dim,self.rope,max_len) for _ in range(n_blocks)])

    def forward(self, x: Tensor, conditioning:Tensor) -> Tensor:
        x = F.normalize(x,p=2,dim=-1) #critical modification
        for block in self.DiT_blocks:
            x=x+block(x, conditioning)
        return x
    