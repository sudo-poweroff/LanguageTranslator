import torch
import torch.nn as nn
import math


class LayerNormalization(nn.Module):

    def __init__(self, features: int, eps:float=10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(features)) # gamma is a learnable parameter
        self.beta = nn.Parameter(torch.zeros(features)) # beta is a learnable parameter

    def forward(self, x):
        # x: (batch, seq_len, hidden_size)
         # Keep the dimension for broadcasting
        mean = x.mean(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # Keep the dimension for broadcasting
        std = x.std(dim = -1, keepdim = True) # (batch, seq_len, 1)
        # eps is to prevent dividing by zero or when std is very small
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class FeedForwardBlock(nn.Module):

    def __init__(self, d_h: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_h, d_ff) # w1 and b1
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_h) # w2 and b2

    def forward(self, x):
        # (batch, seq_len, d_h) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_h)
        x= self.fc1(x)
        x= torch.relu(x)
        x=self.dropout(x)
        x=self.fc2(x)
        return x
    

class ResidualConnection(nn.Module):
    
        def __init__(self, features: int, dropout: float) -> None:
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.norm = LayerNormalization(features)
    
        def forward(self, x, sublayer):
            y = self.norm(x)
            y = sublayer(y)
            y = self.dropout(y)
            return x + y


class ProjectionLayer(nn.Module):

    def __init__(self, d_h, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_h, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_h) --> (batch, seq_len, vocab_size)
        return self.proj(x)