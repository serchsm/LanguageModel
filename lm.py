import argparse

from Models.bi_lstim_lm.BiLSTMLanguageModel import biLSTM_LM

def main(args):
    
    vocabulary_size = 1024
    print(f"model: {args.model}, embedding_size: {args.embedding_size}")
    biLSTM_lm = biLSTM_LM(vocabulary_size, args)
    print(f"embedding_size: {biLSTM_lm.embedding_size}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bi_lstm', help = ' bi_lstm | lstm')
    parser.add_argument('--embedding_size', type=int, default = 256, help = 'embedding size')
    parser.add_argument('--number_layers', type=int, default=2, help='number layers')
    parser.add_argument('--number_hidden, type=int', default=2, help='number hidden units')
    args = parser.parse_args()

    main(args)
