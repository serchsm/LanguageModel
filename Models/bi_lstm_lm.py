import tensorflow as tf


class BiLSTMLanguageModel(tf.keras.Model):

    def __init__(self, vocabulary_size, args):
        super(BiLSTMLanguageModel, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocabulary_size, args.embedding_size)
        self.biLSTM = tf.keras.layers.Bidirectional(
            tf.keras.layers.LSTM(args.number_hidden))
        self.w = tf.keras.layers.Dense(vocabulary_size)

    def call(x_in):
        # Word index to embeddings
        x = self.embedding(x)
        output, state = self.biLSTM(x)
        return self.w(output)

