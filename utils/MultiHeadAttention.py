import torch.nn as nn
import torch

from utils.Head import Head

class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, block_size, channels, num_heads=6, dropout=.2):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, block_size, channels) for i in range(num_heads)])
        self.proj = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out