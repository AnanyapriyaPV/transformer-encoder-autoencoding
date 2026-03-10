import torch
import torch.nn as nn
from attention import MultiHeadSelfAttention
from positional_encoding import PositionalEncoding

class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_heads=4):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.position = PositionalEncoding(embed_dim)

        self.attention = MultiHeadSelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlm_head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = self.position(x)

        attn_out, attn_weights = self.attention(x)
        x = self.norm1(x + attn_out)

        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)

        logits = self.mlm_head(x)
        return logits, attn_weights
