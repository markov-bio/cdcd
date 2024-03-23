import torch
from torch import nn, Tensor
from torch.nn import functional as F

import einops

from .RoPe import RotaryEmbedding
from .attention import SelfAttention


class DiTBlock(nn.Module):
    def __init__(self, embed_dim:int,qkv_dim:int, num_heads:int, cond_dim:int, rope:RotaryEmbedding, max_len=5000):
        super().__init__()
        assert embed_dim>=2*num_heads and embed_dim%num_heads==0, 'the embed_dim must be a multiple of the number of heads'
        self.embed_dim=embed_dim 
        self.qkv_dim=qkv_dim
        self.cond_dim=cond_dim

        self.attention=SelfAttention(embed_dim, qkv_dim, num_heads, rope) 
        self.rope=rope

        self.make_scale_shift=MakeScaleShift(cond_dim, embed_dim)
        self.layernorm1=nn.LayerNorm(torch.broadcast_shapes((embed_dim,)))
        self.layernorm2=nn.LayerNorm(torch.broadcast_shapes((embed_dim,)))

        self.feedforward=nn.Linear(embed_dim,embed_dim) 
        nn.init.zeros_(self.feedforward.weight)
        nn.init.zeros_(self.feedforward.bias)
        
        
    def forward(self,x:torch.Tensor,t_conditioning:torch.Tensor, attn_mask=None)->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor (b, l, c)
            conditioning (torch.Tensor): conditioning (l,). 
            attn_mask (torch.Tensor): masks the [PAD] tokens (b, 1, l, l). 

        Returns:
            torch.Tensor: tensor x.shape
        """
        
        #here we create the scale-shift parameters from the conditioning
        alpha_1,beta_1,gamma_1,alpha_2,beta_2,gamma_2=self.make_scale_shift(t_conditioning)

        res=x.clone()

        x=self.layernorm1(x)
        x=apply_scale_shift(x,gamma_1,beta_1)
        x=self.attention(x,attn_mask)
        x=apply_scale_shift(x,alpha_1)
        x=x+res

        x=self.layernorm2(x)
        x=apply_scale_shift(x,gamma_2,beta_2)
        x=self.feedforward(x)
        x=apply_scale_shift(x,alpha_2)
        
        return x

        
        

class MakeScaleShift(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()

        self.linear=nn.Linear(cond_dim, embed_dim*6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, conditioning:torch.Tensor):
        assert conditioning.dim() == 2, "all of the cells must have the same conditioning"
        return self.linear(conditioning).chunk(6,dim=-1)
 
def apply_scale_shift(x, scale, shift:Tensor=None):

    scale=scale+1
    x=einops.einsum(x,scale,'b ... c, b c -> b ... c')
    
    if shift is not None: 
        x=x+shift.unsqueeze(1)

    return F.layer_norm(x, normalized_shape=(x.shape[-1],))

