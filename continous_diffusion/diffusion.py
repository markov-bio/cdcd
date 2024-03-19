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

class MaskDiffusion(nn.Module):
    def __init__(self, model:nn.Module, loss:nn.Module, conditioning:nn.Module):
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
        noise = t.to(tokens.device)  #Index on cpu then send resulting tensor to cuda

        x=self.embedder(tokens)
        
        standard_deviation_normalizer=torch.sqrt(torch.tensor(self.embedder.embedding_dim)/(noise**2+1))
        x=einops.einsum(x,standard_deviation_normalizer,'b ..., b -> b ...')
        
        return x,noise

    # for Composer, we need this to be called forward()
    def forward(self,x,noise):
        conditioning=self.conditioning(noise) 

        x_0=self.model(x,conditioning)
        return x

    def alphaXscore(self,x,noise):
        #it is called alphaXscore because it returns alpha*score
        embeddings=self.predict(x,noise)

        prob=F.softmax(self.un_embedder(embeddings), dim=-1)

        x_0=self.embedder.expected_embedding(prob)

        return (x-x_0)/noise
    