import numpy as np
import tensorflow as tf

from pathlib import Path

def get_text_corpus():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    with open(Path(path_to_file), mode='r') as fid:
        text = fid.read()
    return text

def generate_windows(window_length, full_sequence):
    return [full_sequence[i:i+window_length] for i in range(0, len(full_sequence)-window_length+1)] 

def split_input_target(window):
    return window[:-1], window[-1:]

def create_batched_inputs_targets(batch_size, windows):
    ds = tf.data.Dataset.from_tensor_slices(windows).shuffle(len(windows))
    return ds.map(split_input_target).batch(batch_size, drop_remainder=True).prefetch(4)

class TextGenerator():
    def __init__(self, lm, number_words, tokenizer, ngram, temperature=1.0):
        self.lm = lm
        self.number_words = number_words
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.ngram = ngram
    
    def next_input(self, in_sequence):
        return in_sequence[-self.ngram+1:]

    def sample_next_word(self, predictions):
        scaled_logits = np.log(predictions)/self.temperature
        return tf.random.categorical(np.array(scaled_logits), num_samples=1).numpy()[0]

    def generate_text(self, next_sequence):
        generated_sequence = next_sequence
        for _ in range(self.number_words):
            yhat = self.lm.predict(next_sequence)
            next_index = self.sample_next_word(yhat)
            generated_sequence = tf.concat([generated_sequence, next_index], axis=-1)
            next_sequence = self.next_input(generated_sequence)
        return self.tokenizer.generate_text(generated_sequence)
