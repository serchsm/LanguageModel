import argparse
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split

from helpers import utilities
from helpers.utilities import TextGenerator
from Tokenizer.tokenizer import TextTokenizer
from Models.language_models import ngram_lstm


def main(args):
    text = utilities.get_text_corpus()
    
    tokenizer = TextTokenizer(text)
    encoded_text = tokenizer.encode_text(text)[0]
    
    windows = utilities.generate_windows(args.ngram, encoded_text)
    train_windows, validation_windows = train_test_split(windows, test_size=0.1, random_state=42)
    seed_sequence = validation_windows[-1:][0][:-1]
    print(f"w_length: {args.ngram}, windows: {windows[:10]}, seed_sequence= {seed_sequence}")
    training_dataset = utilities.create_batched_inputs_targets(args.batch_size, train_windows[:100])
    validation_dataset = utilities.create_batched_inputs_targets(args.batch_size, validation_windows[:10])
    
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units) 
    ngram_lm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    ngram_lm.summary()
    ngram_lm.fit(training_dataset, epochs=args.epochs, validation_data=validation_dataset)

    for temperature in [0.5, 1.0, 1.5, 2.0]:
        generator = TextGenerator(ngram_lm, args.number_words, tokenizer, args.ngram, temperature=temperature)
        print(f"Temperature: {temperature}. Generated Text......")
        print(f"{generator.generate_text(seed_sequence)}")
        print("")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=4, help='Ngram order')
    parser.add_argument('--epochs', type=int, default=2, help='Number epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to train model')
    # parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 32, help = 'embedding size')
    # parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--lstm_units', type=int, default=256, help='number lstm units')
    parser.add_argument('--number_words', type=int, default=100, help='Number words to generate with LM')
    args = parser.parse_args()

    main(args)
