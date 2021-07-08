import tensorflow as tf

from Pathlib import Path

def get_text_corpus():
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    with open(Path(path_to_file), model='r') as fid:
        text = fid.read()
    return text

def generate_windows(window_length, full_sequence):
    return [full_sequence[i:i+window_length] for i in range(0, len(full_sequence)-window_length+1)] 

def split_input_target(window):
    return window[:-1], window[-1:]
