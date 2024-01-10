import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_h: int, vocab_size: int) -> None:
        super().__init__()
        self.d_h = d_h
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_h) #Embedding layer (given a certain token provides always the same embedding)

    def forward(self, x):
        # (batch, seq_len) --> (batch, seq_len, d_h)
        # Multiply by sqrt(d_h) to scale the embeddings according to the paper
        return self.embedding(x) * math.sqrt(self.d_h)
    
class PositionalEncoding(nn.Module):

    def __init__(self, d_h: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_h = d_h
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)
        # Create a matrix of shape (seq_len, d_h)
        positional_encoding_matrix = torch.zeros(seq_len, d_h)
        # Create a vector of shape (seq_len)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1) # (seq_len, 1)
        # Create a vector of shape (d_h) to perform the positional values
        div_term = torch.exp(torch.arange(0, d_h, 2).float() * (-math.log(10000.0) / d_h)) # (d_h / 2)
        # Apply sine to even indices
        positional_encoding_matrix[:, 0::2] = torch.sin(position * div_term) # sin(position * (10000 ** (2i / d_h))
        # Apply cosine to odd indices
        positional_encoding_matrix[:, 1::2] = torch.cos(position * div_term) # cos(position * (10000 ** (2i / d_h))
        # Add a batch dimension to the positional encoding
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0) # (1, seq_len, d_h)
        # Register the positional encoding as a buffer
        self.register_buffer('positional_encoding_matrix', positional_encoding_matrix)

    def forward(self, x):
        x = x + (self.positional_encoding_matrix[:, :x.shape[1], :]).requires_grad_(False) # (batch, seq_len, d_h)
        return self.dropout(x)
