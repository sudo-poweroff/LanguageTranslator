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


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_h: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_h = d_h # Embedding vector size
        self.h = h # Number of heads
        # Make sure d_h is divisible by h
        assert d_h % h == 0, "d_h is not divisible by h"

        self.d_k = d_h // h # Dimension of vector seen by each head
        self.query_layer = nn.Linear(d_h, d_h, bias=False) # query weights
        self.key_layer = nn.Linear(d_h, d_h, bias=False) # key weights
        self.value_layer = nn.Linear(d_h, d_h, bias=False) # Value weights
        self.fc1 = nn.Linear(d_h, d_h, bias=False) # Last matrix
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1] #head dimension
        # Just apply the formula from the paper
        # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            # Write a very low value (indicating -inf) to the positions where mask == 0 therefore the irrelevant words
            attention_scores.masked_fill_(mask == 0, -1e9)
        attention_scores = attention_scores.softmax(dim=-1) # (batch, h, seq_len, seq_len) Apply softmax
        if dropout is not None:
            attention_scores = dropout(attention_scores)
        # (batch, h, seq_len, seq_len) --> (batch, h, seq_len, d_k)
        # return attention scores which can be used for visualization
        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.query_layer(q) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)
        key = self.key_layer(k) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)
        value = self.value_layer(v) # (batch, seq_len, d_h) --> (batch, seq_len, d_h)

        # (batch, seq_len, d_h) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Calculate attention
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        
        # Combine all the heads together
        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_h)
        # contiguous to put the in coniguous memory space and to modify it in place
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k) 

        # Multiply by Wo
        # (batch, seq_len, d_h) --> (batch, seq_len, d_h)  
        return self.fc1(x)


class EncoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
    

class DecoderBlock(nn.Module):

    def __init__(self, features: int, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x
    

class ProjectionLayer(nn.Module):

    def __init__(self, d_h, vocab_size) -> None:
        super().__init__()
        self.proj = nn.Linear(d_h, vocab_size)

    def forward(self, x) -> None:
        # (batch, seq_len, d_h) --> (batch, seq_len, vocab_size)
        return self.proj(x)