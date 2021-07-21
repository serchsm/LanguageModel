import argparse
import numpy as np
import tensorflow as tf

from pathlib import Path
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
    print(f"w_length: {args.ngram}, number train windows: {len(train_windows)}")
    print(f"Number validation windows: {len(validation_windows)}, seed_sequence= {seed_sequence}")
    training_dataset = utilities.create_batched_inputs_targets(args.batch_size, train_windows, is_debug=args.debug)
    validation_dataset = utilities.create_batched_inputs_targets(args.batch_size,
                                                                 validation_windows,
                                                                 is_debug=args.debug)
    
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units) 
    ngram_lm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    ngram_lm.summary()
    ngram_lm.fit(training_dataset, epochs=args.epochs, validation_data=validation_dataset)

    for temperature in [0.5, 1.0, 1.5, 2.0]:
        generator = TextGenerator(ngram_lm,
                                  args.number_words,
                                  tokenizer, args.ngram,
                                  temperature=temperature,
                                  batch_size=args.batch_size)
        print(f"Temperature: {temperature}. Generated Text......")
        print(f"{generator.generate_text(seed_sequence)}")
        print("")

    trained_path = Path('./TrainedModels')
    trained_path.mkdir(exist_ok=True)
    trained_model_filename = trained_path / 'simple_nlm.h5'
    ngram_lm.save(str(trained_model_filename))

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
    parser.add_argument('--debug', type=bool, default=False, help='If True, only use very few samples to train LM')
    args = parser.parse_args()

    main(args)
