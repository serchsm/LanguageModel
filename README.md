# Shakespeare Word Based Language Model

Code creates a neural LMs based on Shakespeare text. It trains a word based LM using n-grams and generates text by 
sampling from the predicted distribution of the next word.  
The LMs can learn its own embeddings or use pre-trained GloVe embeddings. Next, there are two stacked LSTMs layers, and 
the final layer is a dense layer that maps to the output vocabulary (words in corpus).

## Training
The LMs are trained with 90% of all n-grams that can be found in the Shakespeare corpus. Training for 150 epochs using a
GPU (Tesla P-100) and batch size of 512 with each LSTM using 512 units achieved 55% accuracy. The same architecture but 
loading Glove embeddings (dimension of 50) achieved 59% accuracy when trained for 189 epochs.  
Better models require training for more epochs, and the choice of using pre-trained embeddings helps with achieving 
higher accuracy faster.

## Inference
Run the generate_text.py script to generate text. It needs a pretrained model that can be loaded. The supplied model in
this repo is a dummy model trained for a few epochs only and with a tiny fraction of the training data.
