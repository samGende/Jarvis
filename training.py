import torch
from utils.Encoder import Encoder
from utils.BigramLanguageModel import BigramLanguageModel
from utils.BiagramAttentionModel import BigramAttentionModel


## load data from txt file 
with open('easy.txt', 'r') as file:
    text = file.read()

##
encoder = Encoder(text)
encoded_txt = encoder.encode(text)

n_chars = len(encoded_txt)

print(f'{n_chars:,} characters in data set')

## split data in train and test 
n_train = int(n_chars * 0.9)

train = torch.tensor(encoded_txt[:n_train])
test = torch.tensor(encoded_txt[n_train:])

print(f'{len(train):,} characters in training set')
print(f'{len(test):,} characters in test set')

print(f'vocab size: {encoder.vocab_size:,}')

## batch data 
block_size = encoder.vocab_size
batch_size = 64

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train if split == 'train' else test
    ix = torch.randint(len(data) - block_size, (batch_size,)) # get a random value
    x = torch.stack([data[i:i+block_size] for i in ix]) # the first block size (context)
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]) # the target
    return x, y

batch, targets = get_batch('train')
print(f'shape of batch: {batch.shape}')

## do training 
#model = BigramLanguageModel(encoder.vocab_size)
model = BigramAttentionModel(encoder.vocab_size, 256)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
iters = 100

for i in range(iters):
    batch, targets = get_batch('train')
    logits, loss = model(batch, targets)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
 
    if(i % 10 == 0):
        print(f'loss: {loss:f} iterartion: {i}')

context = torch.tensor(encoder.encode('easy'))
context = context.unsqueeze(0)
pred = torch.squeeze(model.genrate(context, 10))
print(pred.shape)
words = encoder.decode(pred.tolist())
print(f'jarvises first words \" {words} \"')

