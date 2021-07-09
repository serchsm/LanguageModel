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

def next_word(predictions, temperature=0.5):
    scaled_logits = np.log(predictions)/temperature
    return tf.random.categorical(np.array(scaled_logits), num_samples=1).numpy()[0]

def next_input(in_sequence, ngram_order):
    return in_sequence[-ngram_order+1:]
