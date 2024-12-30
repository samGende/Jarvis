from idna import encode
import torch

from utils.Encoder import Encoder 
from utils.BigramLanguageModel import BigramLanguageModel

text = 'abcH defdgqooureutoirwpok.,m xa,mrdsourewpirpq;l;d,ca;pweiro,z,mmfofpipwptqutyuiowqasdfghjklzxcvbnm,pqiwoeruigjqlv.,lerioewutiowpeoriQWERTYUIOPASDFGHJKKLZXCVBNM'

encoder = Encoder(text)
print("chars to index: ")
print(encoder.chars_to_index)
print("index to chars")
print(encoder.index_to_chars)

encoding = encoder.encode('Hello Wrld')
print(encoding)

decoded_string = encoder.decode(encoding)
print(decoded_string)

model = BigramLanguageModel(encoder.vocab_size)

encoding = torch.tensor(encoding)

logits, loss = model(encoding)


encoded_pred = model.genrate(encoding, 5)
encoded_pred = torch.squeeze(encoded_pred)
encoded_pred = [int(num) for num in encoded_pred]
generation = encoder.decode(encoded_pred)
print(generation)
