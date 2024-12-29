
class Encoder():
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab_size = len(chars)
        self.chars_to_index = {char: index for index, char in enumerate(chars)}
        self.index_to_chars = {index: char for index, char in enumerate(chars)}

    def encode(self, s):
        encoded_list = [self.chars_to_index[c] for c in s]
        return encoded_list
    
    def decode(self, l):
        decoded_string = ''.join([self.index_to_chars[num] for num in l])
        return decoded_string
