import torch
from torch import nn, Tensor
from torch.nn import functional as F
import math

from .DiT_block import DiTBlock
# taken from https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class TransformerModel(nn.Module):

    def __init__(self, embed_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        self.pos_encoder = PositionalEncoding(embed_dim)

        self.DiT_blocks=nn.Sequential(*[DiTBlock(embed_dim,num_heads,cond_dim,max_len) for _ in range(n_blocks)])

    def forward(self, x: Tensor, conditioning:Tensor) -> Tensor:
        x = F.normalize(x,p=2,dim=-1) #critical modification
        x = self.pos_encoder(x)
        for block in self.DiT_blocks:
            x=x+block(x, conditioning)
        return x
    
    
class PositionalEncoding(nn.Module):

    def __init__(self, embed_dim: int,  max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        x = x + self.pe[:,:x.size(1)]
        return x