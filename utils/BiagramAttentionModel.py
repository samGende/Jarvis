import torch.nn as nn
import torch.nn.functional as F
import torch

from utils.Block import Block

class BigramAttentionModel(nn.Module):
    def __init__(self, vocab_size, block_size, n_embedding=384, n_head=6, n_layers=6):
      super().__init__()
      self.token_embeding = nn.Embedding(vocab_size, n_embedding)
      self.pos_embeding = nn.Embedding(block_size, n_embedding)
      self.blocks = nn.Sequential(*[Block(n_embedding, n_head) for i in range(n_layers)])
      self.final_ln = nn.LayerNorm(n_embedding)
      self.lm_head = nn.Linear(n_embedding, vocab_size)

    def forward(self, context, targets=None):
      B,T = context.shape
      tok_emb = self.token_embeding(context)
      pos_emb = self.pos_embeding(torch.arange(T))
      x = tok_emb + pos_emb
      x = self.blocks(x)
      x = self.final_ln(x)
      logits = self.lm_head(x)

      if targets is None:
        loss = None
      else:
        B,T,C = logits.shape
        logits = logits.view(B*T, C)
        targets = targets.view(B*T)
        loss = F.cross_entropy(logits, targets)
      return logits, loss
    
    def genrate(self, context, max_new_tokens):
      for i in range(max_new_tokens):
        logits, _ = self(context)
        if(logits.dim() == 2):
          logits = logits.unsqueeze(0)
        logits = logits[:,-1, :]
        probs = F.softmax(logits, dim=1)
        pred = torch.multinomial(probs, num_samples=1)
        if(context.dim() ==1):
          context = context.unsqueeze(0)
        context = torch.cat((context, pred), dim=1)
      return context
