import tensorflow as tf


class GRULM(tf.keras.Model):
    def __init__(self, vocabulary_size, embedding_size=128, number_hidden=256):
        super(GRULM, self).__init__()
        self.embedding = tf.keras.layers.Embedding(vocabulary_size, embedding_size)
        self.gru = tf.keras.layers.GRU(number_hidden)
        self.fc = tf.keras.layers.Dense(vocabulary_size)
    
    def call(self, x_in):
       # Word index to embedding
       x = self.embedding(x_in)
       print(f"emb shape: {x.shape}")
       output = self.gru(x)
       return self.fc(output)
