import torch.nn as nn

from utils.FeedForward import FeedForward
from utils.MultiHeadAttention import MultiHeadAttention

class Block(nn.Module):
    def __init__(self, n_embedding, n_head, block_size = 256):
        super().__init__()
        head_size = n_embedding // n_head
        self.sa = MultiHeadAttention(head_size, block_size, n_embedding)
        self.ffwd = FeedForward(n_embedding)
        self.ln1 = nn.LayerNorm(n_embedding)
        self.ln2 = nn.LayerNorm(n_embedding)


    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x