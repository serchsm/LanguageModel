import argparse
import numpy as np
import tensorflow as tf

# import utilities as utils
from helpers import utilities

# from Models.bi_lstm_lm import GRULM
# from Processor.data_processor import get_data

def main(args):
    text = utilities.get_text_corpus()
    print(f"text: {text[:100]}")
    # vocabulary_size = 1024
    # gru_lm = GRULM(vocabulary_size)
    # for i, mini_batch in enumerate(dataset):
    #     print(f"shape: {mini_batch.shape}")
    #     output = gru_lm(mini_batch)
    #     break

    # print(f"output shape: {output.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 256, help = 'embedding size')
    parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--number_hidden', type=int, default=512, help='number hidden units')
    args = parser.parse_args()

    main(args)
