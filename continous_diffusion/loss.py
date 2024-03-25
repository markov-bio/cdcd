import torch
import torch.nn as nn
import einops

from .embedding import Embedder, UnEmbedder
from .scheduling import AdaptiveSchedule

class Loss(nn.Module):
    def __init__(self, embedder: Embedder, noise_schedule: AdaptiveSchedule):
        super().__init__()
        self.embedder = embedder
        self.un_embedder = UnEmbedder(embedder)
        self.noise_schedule = noise_schedule
        
        self.cross_entropy_loss = nn.CrossEntropyLoss(reduction='none')

    def forward(self, target_tokens: torch.Tensor, output_embeddings: torch.Tensor, sigma: float) -> torch.Tensor:
        # Transform the output embeddings back to logits
        logits = self.un_embedder(output_embeddings)
        logits = einops.rearrange(logits, 'b ... c -> b c (...)')

        # Flatten target tokens 
        target_tokens = target_tokens.flatten(start_dim=1)

        # Compute cross-entropy loss
        loss = self.cross_entropy_loss(logits, target_tokens)
        
        #averaging over the non-padding tokens
        padding_mask = target_tokens == (self.embedder.num_embeddings-1)
        loss[padding_mask] = 0
        loss = loss.sum(dim=-1) / (padding_mask.shape[-1] - padding_mask.float().sum(dim=-1))

        # Update the adaptive schedule with the current loss and sigma (useful for plotting)
        self.noise_schedule.add_data(loss, sigma)

        return loss.mean()
