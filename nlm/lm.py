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
    seed_sequence = validation_windows[-1:][0][:-1]
    print(f"w_length: {args.ngram}, windows: {windows[:10]}, seed_sequence= {seed_sequence}")
    training_dataset = utilities.create_batched_inputs_targets(args.batch_size, train_windows[:100])
    validation_dataset = utilities.create_batched_inputs_targets(args.batch_size, validation_windows[:10])
    
    vocabulary_size = len(tokenizer.tokenizer.word_index)
    ngram_lm = ngram_lstm(vocabulary_size, args.embedding_size, args.lstm_units) 
    ngram_lm.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['acc'])
    ngram_lm.summary()
    ngram_lm.fit(training_dataset, epochs=args.epochs, validation_data=validation_dataset)

    number_words = 10
    generated_sequence = seed_sequence
    next_sequence = seed_sequence
    for _ in range(number_words):
        yhat = ngram_lm.predict(next_sequence)
        next_word_index = utilities.next_word(yhat, temperature=0.9)
        generated_sequence = tf.concat([generated_sequence, next_word_index], axis=-1)
        print(f"in_seq: {next_sequence}, next_ind: {next_word_index}, gen_seq: {generated_sequence}")
        next_sequence = utilities.next_input(generated_sequence, args.ngram)
        print(f"next iter seq: {next_sequence}")
    print(f"Generated Text: {tokenizer.generate_text(generated_sequence)}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ngram', type=int, default=4, help='Ngram order')
    parser.add_argument('--epochs', type=int, default=2, help='Number epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size to train model')
    # parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 32, help = 'embedding size')
    # parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--lstm_units', type=int, default=256, help='number lstm units')
    args = parser.parse_args()

    main(args)
