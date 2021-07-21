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
    return np.asarray([full_sequence[i:i + window_length] for i in range(0, len(full_sequence) - window_length + 1)])


def split_input_target(window):
    return window[:-1], window[-1:]


def create_batched_inputs_targets(batch_size, windows, is_debug=False):
    if is_debug:
        windows = windows[:1000]
    ds = tf.data.Dataset.from_tensor_slices(windows).shuffle(len(windows))
    return ds.map(split_input_target).batch(batch_size, drop_remainder=True).prefetch(4)


class TextGenerator():
    def __init__(self, lm, number_words, tokenizer, ngram, temperature=1.0, batch_size=4):
        self.lm = lm
        self.number_words = number_words
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.ngram = ngram
        self.batch_size = batch_size

    def next_input(self, in_sequence):
        return in_sequence[-self.ngram + 1:]

    def sample_next_word(self, predictions):
        scaled_logits = np.log(predictions) / self.temperature
        new_distribution = np.exp(scaled_logits)
        new_distribution /= np.sum(new_distribution, axis=-1, keepdims=True)
        return np.asarray([np.random.choice(new_distribution.shape[1], p=new_distribution[0])])

    def generate_batch(self, x_in):
        return np.tile(x_in.reshape((1, self.ngram-1)), [self.batch_size, 1])

    def generate_text(self, next_sequence):
        generated_sequence = next_sequence
        for _ in range(self.number_words):
            # Bug due to not implementing batched input
            yhat = self.lm.predict(self.generate_batch(next_sequence))
            next_index = self.sample_next_word(yhat)
            generated_sequence = np.concatenate([generated_sequence, next_index])
            next_sequence = self.next_input(generated_sequence)
        return self.tokenizer.generate_text(generated_sequence)[0]
