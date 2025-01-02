import torch.nn as nn
import torch.nn.functional as F
import torch

class Head(nn.Module):
    def __init__(self, head_size, block_size, channels, dropout =.2):
        super().__init__()
        self.key = nn.Linear(channels, head_size, bias=False)
        self.query = nn.Linear(channels, head_size, bias=False)
        self.value = nn.Linear(channels, head_size, bias=False)
        self.lower_tri = torch.tril(torch.ones(block_size, block_size))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        attent = q @ k.transpose(-2, -1)
        attent = attent.masked_fill(self.lower_tri[:T, :T] == 0, float('-inf'))
        attent = F.softmax(attent, dim=1)
        attent = self.dropout(attent)
        return attent @ v
