import numpy as np

from tensorflow.keras import preprocessing


class TextTokenizer():
    def __init__(self, input_text):
        self.tokenizer = preprocessing.text.Tokenizer()
        self.tokenizer.fit_on_texts([input_text])
    
    def encode_text(self, text):
        return np.array(self.tokenizer.texts_to_sequences([text])) - 1

    def generate_text(self, in_sequence):
        return self.tokenizer.sequences_to_texts(in_sequence.reshape([1,-1])+1)
