import torch
from torch import nn

from .model import TransformerModel



import einops
from tqdm import tqdm
import torch 
from torch.nn import functional as F
import numpy as np

from .scheduling import AdaptiveSchedule
from .loss import Loss
from .conditioning import TimeConditioning
from .model import TransformerModel

class Diffusion(nn.Module):
    def __init__(self, model:TransformerModel, loss:Loss, conditioning:TimeConditioning):
        super().__init__()

        self.model=model

        self.loss=loss
        self.embedder= loss.embedder
        self.un_embedder=loss.un_embedder
        self.schedule=loss.schedule
        
        self.conditioning=conditioning

        self.n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
    def make_sample(self,tokens:torch.Tensor):

        t = self.schedule.sample(shape=(tokens.shape[0],))
        sigma = t.to(tokens.device)  #Index on cpu then send resulting tensor to cuda

        x=self.embedder(tokens) 
        x=x + einops.einsum(torch.randn_like(x), sigma, 'b ..., b -> b ...') 
        standard_deviation_normalizer=torch.sqrt(torch.tensor(self.embedder.embed_dim)/(sigma**2+1))
        x=einops.einsum(x,standard_deviation_normalizer,'b ..., b -> b ...')
        
        return x,sigma

    # for Composer, we need this to be called forward()
    def forward(self,x,sigma):
        conditioning=self.conditioning(sigma) 

        x_0=self.model(x,conditioning)
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

        Args:

        """
        shape=(batch_size,sequence_lenght,self.embedder.embed_dim) 
        
        x = torch.randn(shape, device=device) * self.schedule.tmax
        
        return self.denoise(x,self.schedule.tmax,n_steps,device)
    
    @torch.no_grad()
    def denoise(self, x, noise_level, n_steps, device='cpu'):

        timesteps=self.schedule.make_timesteps(n_steps,tmax=noise_level,device=device).unsqueeze(1)

        sequence_lenght=x.shape[2]
        for i in tqdm(range(n_steps-1)):        
            
            delta_x = self.alphaXscore(x, timesteps[i])
            delta_t=timesteps[i+1]-timesteps[i]
            x = x - delta_x * delta_t
            
            assert torch.any(torch.isnan(x)).item() is False, 'generator output is NaN'
        
        return x 