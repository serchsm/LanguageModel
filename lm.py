import argparse

from Models.bi_lstm_lm import BiLSTMLanguageModel as biLSTM_LM

def main(args):
    vocabulary_size = 1024
    print(f"model: {args.model}, embedding_size: {args.embedding_size}")
    biLSTM_lm = biLSTM_LM(vocabulary_size, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 256, help = 'embedding size')
    parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--number_hidden', type=int, default=512, help='number hidden units')
    args = parser.parse_args()

    main(args)
