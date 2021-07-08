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
    train_windows, validation_windows = train_test_split(windows, test_size=0.1, random_state=42)
    print(f"w_length: {args.ngram}, windows: {windows[:10]}")
    
    training_dataset = utilities.create_batched_inputs_targets(args.batch_size, train_windows[:100])
    validation_dataset = utilities.create_batched_inputs_targets(args.batch_size, validation_windows[:10])
    
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units) 
    ngram_lm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    ngram_lm.summary()
    ngram_lm.fit(training_dataset, epochs=args.epochs, validation_data=validation_dataset)

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
    parser.add_argument('--epochs', type=int, default=20, help='Number epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to train model')
    # parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 32, help = 'embedding size')
    # parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--lstm_units', type=int, default=256, help='number lstm units')
    args = parser.parse_args()

    main(args)
