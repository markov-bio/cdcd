import einops
import torch
from torch import nn, Tensor

from .conditioning import TimeConditioning
from .DiT_block import DiTBlock
from .RoPe import RotaryEmbedding

class DiffusionTransformer(nn.Module):
    def __init__(self, embed_dim, qkv_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        # ensure the embedding dimension is compatible with the number of heads
        assert embed_dim % num_heads == 0 and embed_dim != num_heads, "embedding dimension must be divisible by number of heads and not equal to it."

        self.rope = RotaryEmbedding(qkv_dim // num_heads)
        self.time_conditioning = TimeConditioning(cond_dim, cond_dim)
        self.dit_blocks = nn.Sequential(
            *[DiTBlock(embed_dim, qkv_dim, num_heads, cond_dim, self.rope, max_len) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor, sigma: Tensor, attn_mask: Tensor = None) -> Tensor:
        # clone the input to use later for residual connection
        res = x.clone()
        conditioning = self.time_conditioning(sigma)
        attn_mask = transform_attn_mask(attn_mask)

        # apply the sequence of dit blocks
        for block in self.dit_blocks:
            x = x + block(x, conditioning, attn_mask)
        
        # combine the residuals and the transformed input
        return bmult(res,1-torch.tanh(sigma))+bmult(x,torch.tanh(sigma))

def transform_attn_mask(attn_mask):
    """Transform the attention mask for broadcasting."""
    return einops.einsum(attn_mask,attn_mask, 'b l, b m -> b l m').unsqueeze(1)

def bmult(a,b):
    return einops.einsum(a,b,'b ..., b -> b ...')