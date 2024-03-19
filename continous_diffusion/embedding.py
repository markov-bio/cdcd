import einops
import torch
from torch import nn
from torch.nn import functional as F


class Embedder(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding_dim=embedding_dim
    
    def forward(self, x):
        # Get embeddings
        embeddings = self.embedding(x)
        # Normalize embeddings to have L2 norm = 1
        return F.normalize(embeddings, p=2, dim=-1)

    def expected_embedding(self, x):
        out = F.linear(x, self.weight.t())
        return F.normalize(out,p=2,dim=-1)
    
    @property
    def weight(self) -> torch.Tensor:
        # Return normalized weights as a property
        return F.normalize(self.embedding.weight, p=2, dim=-1)
        

class UnEmbedder(nn.Module):
    def __init__(self, normalized_embedder: Embedder):
        super().__init__()
        assert isinstance(normalized_embedder,Embedder), "the embedding needs to be normalized"
        self.normalized_embedder = normalized_embedder

    def forward(self, x):
        return F.linear(x, self.weight)

    @property
    def weight(self) -> torch.Tensor:
        # Return normalized weights as a property
        return self.normalized_embedder.weight
    