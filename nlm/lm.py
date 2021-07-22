import argparse
import numpy as np
import tensorflow as tf

from pathlib import Path
from sklearn.model_selection import train_test_split

from helpers import utilities
from helpers.utilities import TextGenerator
from Embeddings.embeddings import GloveEmbeddings
from Models.language_models import ngram_lstm, ngram_lstm_with_pretrained_embeddings
from Tokenizer.tokenizer import TextTokenizer


def main(args):
    text = utilities.get_text_corpus()
    
    tokenizer = TextTokenizer(text)
    encoded_text = tokenizer.encode_text(text)[0]
    
    windows = utilities.generate_windows(args.ngram, encoded_text)
    train_windows, hold_out_windows = train_test_split(windows, test_size=0.1, random_state=42)
    seed_sequence = hold_out_windows[-1:][0][:-1]
    print(f"w_length: {args.ngram}, number train windows: {len(train_windows)}")
    print(f"Number hold out windows: {len(hold_out_windows)}, seed_sequence= {seed_sequence}")
    training_dataset = utilities.create_batched_inputs_targets(args.batch_size, train_windows, is_debug=args.debug)
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
    if args.model == 'ngram_lstm':
        ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units)
    else:
        glove = GloveEmbeddings("http://nlp.stanford.edu/data/glove.6B.zip", Path("./Embeddings/glove.6B.zip"), 50)
        glove.get_embedding()
        # ngram_lm = ngram_lstm_with_pretrained_embeddings(vocabulary_size, embedding_matrix, args.lstm_units)
    ngram_lm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    ngram_lm.summary()
    ngram_lm.fit(training_dataset, epochs=args.epochs, callbacks=[early_stopping_cb])

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
    parser.add_argument('--model', type=str, default='ngram_lstm', help = "'ngram_lstm' or 'pretrained_embeddings_lstm'")
    parser.add_argument('--embedding_size', type=int, default = 32, help = 'embedding size')
    parser.add_argument('--lstm_units', type=int, default=256, help='number lstm units')
    parser.add_argument('--number_words', type=int, default=100, help='Number words to generate with LM')
    parser.add_argument('--debug', type=bool, default=False, help='If True, only use very few samples to train LM')
    args = parser.parse_args()

    main(args)
