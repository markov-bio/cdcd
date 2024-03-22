import torch
from torch import nn
import einops

from .embedding import Embedder,UnEmbedder
from .scheduling import AdaptiveSchedule


class Loss(nn.Module):
    def __init__(self, embedder:Embedder, schedule:AdaptiveSchedule):
        super().__init__()

        self.embedder=embedder
        self.un_embedder=UnEmbedder(embedder)
        self.schedule=schedule
        
        self.ce=nn.CrossEntropyLoss(reduction='none',ignore_index=embedder.num_embeddings-1)
        

    def forward(self,target_tokens:torch.Tensor, output_embeddings:torch.Tensor, sigma:float):
        logits=self.un_embedder(output_embeddings)
       
        logits=einops.rearrange(logits,'b ... c -> b c (...)')
        ce_loss=self.ce(logits,target_tokens.flatten(start_dim=1)).mean(dim=-1)
       
        self.schedule.add_data(ce_loss,sigma)
        
        return ce_loss.mean() 