import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, n_embedding, dropout = .2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embedding, 4*n_embedding),
            nn.ReLU(),
            nn.Linear(4*n_embedding, n_embedding),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.layers(x)