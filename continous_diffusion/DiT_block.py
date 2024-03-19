import torch
from torch import nn
from torch.nn import functional as F

import einops

class MakeScaleShift(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()

        self.linear=nn.Linear(cond_dim, embed_dim*6)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, conditioning:torch.Tensor):
        assert conditioning.dim() == 2, "all of the cells must have the same conditioning"
        return self.linear(conditioning).chunk(6,dim=-1)
 
def apply_scale_shift(x, scale, shift=None):

    scale=scale+1
    x=einops.einsum(x,scale,'b ... c, b c -> b ... c')
    
    if shift is not None: 
        new_shape=[1]*x.dim()
        new_shape[0],new_shape[-1]=shift.shape[0],shift.shape[-1]
        shift=shift.view(new_shape)
        x=x+shift

    return F.layer_norm(x, normalized_shape=(x.shape[-1],))



class SelfAttention(nn.Module):
    def __init__(self,embed_dim,num_heads):
        super().__init__()

        self.embed_dim=embed_dim
        self.num_heads=num_heads
        self.linear=nn.Linear(embed_dim,3*embed_dim) #this can be generalized
    
    def forward(self,x):

        x=self.linear(x)
        x=einops.rearrange(x,'... l (h c) -> ... h l c', h=self.num_heads)       
        q,k,v=x.chunk(3,dim=-1)
        

        q=F.normalize(q,p=2,dim=-1)
        k=F.normalize(k,p=2,dim=-1)

        x=F.scaled_dot_product_attention(q,k,v, scale=1)

        x=einops.rearrange(x,'... h l c -> ... l (h c)')
        return x



class DiTBlock(nn.Module):
    def __init__(self, embed_dim:int, num_heads, cond_dim, max_len=5000):
        super().__init__()
        assert embed_dim>=2*num_heads and embed_dim%num_heads==0, 'the embed_dim must be a multiple of the number of heads'
        self.cond_dim=cond_dim
        self.make_scale_shift=MakeScaleShift(cond_dim, embed_dim)

        self.embed_dim=embed_dim 
        self.layernorm1=nn.LayerNorm(torch.broadcast_shapes((embed_dim,)))
        self.attention=SelfAttention(embed_dim, num_heads) 
        
        self.feedforward=nn.Linear(embed_dim,embed_dim) 
        self.layernorm2=nn.LayerNorm(torch.broadcast_shapes((embed_dim,)))

    def forward(self,x:torch.Tensor,conditioning:torch.Tensor|None=None)->torch.Tensor:
        """
        Args:
            x (torch.Tensor): input tensor (b, g, l, c) or (b, l, c)
            conditioning (torch.Tensor, optional): conditioning (l,). Defaults to None

        Returns:
            torch.Tensor: tensor x.shape
        """
        if conditioning is None: 
            conditioning=torch.zeros(x.shape[0],self.cond_dim, device=x.device)
        alpha_1,beta_1,gamma_1,alpha_2,beta_2,gamma_2=self.make_scale_shift(conditioning)

        res=x.clone()

        x=self.layernorm1(x)
        x=apply_scale_shift(x,gamma_1,beta_1)
        x=self.attention(x)
        x=apply_scale_shift(x,alpha_1)
        x=x+res

        x=self.layernorm2(x)
        x=apply_scale_shift(x,gamma_2,beta_2)
        x=self.feedforward(x)
        x=apply_scale_shift(x,alpha_2)
        
        return x