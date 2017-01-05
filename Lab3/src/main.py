import numpy as np
import dataset as ds
import recurrent_neural_network as rnn

def main(path='data/selected_conversations_shorter.txt'):
    np.random.seed(127)
    dataset = ds.Dataset(path)
    rnn.run_language_model(dataset, max_epochs=10000)

if __name__ == '__main__':
    main()
