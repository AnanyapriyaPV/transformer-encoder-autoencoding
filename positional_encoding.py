import torch
import math

class PositionalEncoding(torch.nn.Module):
    def __init__(self, embed_dim, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)

        for pos in range(max_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_dim)))

        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
