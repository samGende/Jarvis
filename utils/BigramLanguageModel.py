import torch.nn as nn
import torch.nn.functional as F
import torch

class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
      super().__init__()
      self.token_embeding = nn.Embedding(vocab_size, vocab_size)

    def forward(self, context, targets=None):
      logits = self.token_embeding(context)
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
