import tensorflow as tf


def ngram_lstm(vocabulary_size, embedding_size=32, lstm_units=128):
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocabulary_size, embedding_size),
        tf.keras.layers.LSTM(lstm_units),
        tf.keras.layers.Dense(vocabulary_size, activation='softmax'),
    ])

def ngram_lstm_with_pretrained_embeddings(vocabulary_size, embedding_matrix, lstm_units=128):
    embedding_values = tf.constant_initializer(embedding_matrix)
    return tf.keras.models.Sequential([
        tf.keras.layers.Embedding(vocabulary_size,
                                  embedding_matrix.shape[1],
                                  embeddings_initializer=embedding_values,
                                  trainable=False),
        tf.keras.layers.LSTM(lstm_units,
                             dropout=0.2,
                             recurrent_dropout=0.2,
                             return_sequences=True),
        tf.keras.layers.LSTM(lstm_units,
                             dropout=0.2,
                             recurrent_dropout=0.2),
        tf.keras.layers.Dense(vocabulary_size, activation='softmax'),
    ])