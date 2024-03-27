import torch
from torch import nn
from torch.nn import functional as F


from tqdm import tqdm
import math


from .model import DiffusionTransformer
from .scheduling import  CauchySchedule
from .loss import Loss
from .model import DiffusionTransformer
from .utils import bmult
from .embedding import Embedder

class Diffusion(nn.Module):
    def __init__(self, model:DiffusionTransformer, loss:Loss):
        super().__init__()

        self.model=model

        self.loss=loss
        self.embedder= loss.embedder
        self.un_embedder=loss.un_embedder
        self.noise_schedule=loss.noise_schedule

        self.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def make_sample(self,tokens:torch.Tensor):
        attn_mask=tokens!=self.embedder.num_embeddings

        t = self.noise_schedule.sample(shape=(tokens.shape[0],))
        sigma = t.to(tokens.device)  #Index on cpu then send resulting tensor to cuda

        x = self.embedder(tokens) * math.sqrt(self.embedder.embed_dim)
        x = x + bmult(torch.randn_like(x), sigma) 
        
        return x,sigma,attn_mask

    def forward(self,x,sigma,attn_mask=None):
        x_0=self.model(x,sigma,attn_mask)
        return x_0

    def alphaXscore(self,x,sigma):
        #it is called alphaXscore because it returns alpha*score
        embeddings=self(x,sigma)

        prob=F.softmax(self.un_embedder(embeddings), dim=-1)

        x_0=self.embedder.expected_embedding(prob)

        return (x-x_0)/sigma
    
    @torch.no_grad()
    def generate(self, batch_size, sequence_lenght, n_steps, device='cpu'):
        """
        It denoises the embedded input x for n_steps starting from t_max to t_min
        """
        shape=(batch_size,sequence_lenght,self.embedder.embed_dim) 
        
        x = torch.randn(shape, device=device) * self.noise_schedule.tmax
        
        return self.denoise(x,self.noise_schedule.tmax,n_steps,device)
    
    @torch.no_grad()
    def denoise(self, x, noise_level, n_steps, device='cpu'):

        timesteps=self.noise_schedule.make_timesteps(n_steps,tmax=noise_level,device=device).unsqueeze(1)

        for i in tqdm(range(n_steps-1)):        
            
            delta_x = self.alphaXscore(x, timesteps[i])
            delta_t=timesteps[i+1]-timesteps[i]
            x = x + delta_x * delta_t
        
        return x 

        

        
class DiffusionModel(Diffusion):

    def __init__(self, 
        embed_dim,
        qkv_dim,
        num_heads,
        cond_dim,
        n_blocks,
        vocab_size,
        device='cpu'
        ):
        dit=DiffusionTransformer(embed_dim,qkv_dim,num_heads,cond_dim,n_blocks)
        embedder=Embedder(vocab_size,embed_dim)
        schedule=CauchySchedule(0.01,200, mu=3.0171,  sigma=1.7785,  height=6.5539, offset=-1.2272)
        loss=Loss(embedder,schedule)
        super().__init__(dit,loss)
        self.to(device)