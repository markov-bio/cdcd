import einops
import torch
from torch import nn, Tensor

from .conditioning import TimeConditioning
from .DiT_block import DiTBlock
from .RoPe import RotaryEmbedding
from .utils import bmult
class DiffusionTransformer(nn.Module):
    def __init__(self, embed_dim, qkv_dim, num_heads, cond_dim, n_blocks, max_len=5000):
        super().__init__()
        # ensure the embedding dimension is compatible with the number of heads
        assert qkv_dim % num_heads == 0 and qkv_dim != num_heads, "embedding dimension must be divisible by number of heads and not equal to it."

        self.rope = RotaryEmbedding(qkv_dim // num_heads)

        self.time_conditioning = TimeConditioning(cond_dim, cond_dim)

        self.dit_blocks = nn.Sequential(
            *[DiTBlock(embed_dim, qkv_dim, num_heads, cond_dim, self.rope, max_len) for _ in range(n_blocks)]
        )

    def forward(self, x: Tensor, sigma: Tensor, attn_mask: Tensor = None) -> Tensor:
        # clone the input to use later for skip connection   
        skip = x.clone()

        # define the preconditioning
        c_noise=torch.log(sigma)/4
        c_skip= 1/(1+sigma**2)
        c_in=torch.sqrt(1/(1+sigma**2))
        c_out = sigma/torch.sqrt(1+sigma**2)
        
        x = bmult(x,c_in)
        conditioning = self.time_conditioning(c_noise)

        attn_mask = transform_attn_mask(attn_mask)

        # apply the sequence of dit blocks
        for block in self.dit_blocks:
            x = x + block(x, conditioning, attn_mask)
        
        # add the precoditioning and the skip connection
        return bmult(skip,c_skip)+bmult(x,c_out)

def transform_attn_mask(attn_mask):
    """Transform the attention mask for broadcasting."""
    if attn_mask==None: return None
    return einops.einsum(attn_mask,attn_mask, 'b l, b m -> b l m').unsqueeze(1)
