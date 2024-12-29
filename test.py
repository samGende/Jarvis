from utils.Encoder import Encoder 

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