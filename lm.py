import argparse
import numpy as np
import tensorflow as tf

from tensorflow.keras.datasets import imdb

from Models.bi_lstm_lm import GRULM
# from Processor.data_processor import get_data

def download_and_pad_imdb(vocabulary_size=1024, max_sequence_length=30):
    (x_train, _), (x_test, _) = imdb.load_data(num_words=vocabulary_size,
                                               maxlen=max_sequence_length)
    return tf.keras.preprocessing.sequence.pad_sequences(x_train)


def get_data(x_in, BATCH_SIZE=64, BUFFER_SIZE=10000):
    dataset = tf.data.Dataset.from_tensor_slices(x_in)
    return dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

def main(args):
    x_in = download_and_pad_imdb()
    dataset = get_data(x_in)
    
    vocabulary_size = 1024
    gru_lm = GRULM(vocabulary_size)
    for i, mini_batch in enumerate(dataset):
        print(f"shape: {mini_batch.shape}")
        output = gru_lm(mini_batch)
        break

    print(f"output shape: {output.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 256, help = 'embedding size')
    parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--number_hidden', type=int, default=512, help='number hidden units')
    args = parser.parse_args()

    main(args)
