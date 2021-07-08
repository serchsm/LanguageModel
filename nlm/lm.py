import argparse
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from helpers import utilities
from Tokenizer.tokenizer import TextTokenizer
from Models.language_models import ngram_lstm

def main(args):
    text = utilities.get_text_corpus()
    
    tokenizer = TextTokenizer(text)
    encoded_text = tokenizer.encode_text(text)[0]
    
    windows = utilities.generate_windows(args.ngram, encoded_text)
    print(f"w_length: {args.ngram}, windows: {windows[:10]}")
    batched_dataset = utilities.create_batched_inputs_targets(args.batch_size, windows)
    for x, y in batched_dataset.take(1):
        print(f"x: {x}, y: {y}")
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units) 
    ngram_lm.summary()


    # vocabulary_size = 1024
    # gru_lm = GRULM(vocabulary_size)
    # for i, mini_batch in enumerate(dataset):
    #     print(f"shape: {mini_batch.shape}")
    #     output = gru_lm(mini_batch)
    #     break

    # print(f"output shape: {output.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=4, help='Ngram order')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to train model')
    # parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 32, help = 'embedding size')
    # parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--lstm_units', type=int, default=256, help='number lstm units')
    args = parser.parse_args()

    main(args)
