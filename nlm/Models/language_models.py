import tensorflow as tf


def ngram_lstm(vocabulary_size, embedding_size=32, lstm_units=128):
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_size),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dense(vocabulary_size, activation='softmax'),
    ])

class LSTMLM(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size=128, number_hidden=256):
        super(LSTMLM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.lstm= tf.keras.layers.LSTM(number_hidden)
        self.fc = tf.keras.layers.Dense(vocabulary_size)
    
    def call(self, x_in):
       # Word index to embedding
       x = self.embedding(x_in)
       print(f"emb shape: {x.shape}")
       output = self.lstm(x)
       return self.fc(output)
