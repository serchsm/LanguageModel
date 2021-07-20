import argparse
import numpy as np
import tensorflow as tf

from Tokenizer.tokenizer import TextTokenizer
from helpers import utilities
from helpers.utilities import TextGenerator

trained_models = {'simple': './TrainedModels/simple_nlm.h5'}

def main(args):
   print(f"Loading {args.model_type} model...")
   nlm = tf.keras.models.load_model(trained_models[args.model_type])

   text = utilities.get_text_corpus()
   tokenizer = TextTokenizer(input_text=text)
   encoded_text = tokenizer.encode_text(text)[0]
   windows = utilities.generate_windows(args.ngram_order, encoded_text)
   seed_index = np.random.choice(len(windows))

   for temperature in [0.1, 0.5, 1.0, 1.5, 2.0, 5.0]:
      tg = TextGenerator(nlm, 50, tokenizer, args.ngram_order, temperature=temperature)
      print(f"Temperature: {temperature}. Generated Text......")
      print(f"{tg.generate_text(windows[seed_index])}")
      print("")




if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument("--model_type", type=str, default='simple', help="Type of model to load. One of 'simple', ")
   parser.add_argument("--ngram_order", type=int, default=4, help="Order of trained LM")
   main(parser.parse_args())