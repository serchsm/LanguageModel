import tensorflow as tf

class BiLSTMLanguageModel(tf.keras.Model):

    def __init__(self, vocabulary_size, args):
        super(BiLSTMLanguageModel, self).__init__()
        self.embbeding_size = args.embedding_size
        

