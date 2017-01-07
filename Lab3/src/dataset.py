import numpy as np
import collections
from itertools import tee

class Dataset:
    def __init__(self, input_file):
        with open(input_file, 'r') as f:
            data = f.read()

        occurences = collections.Counter(data)
        self.sorted_chars = sorted(occurences.keys(), key=occurences.get, reverse=True)

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))

        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}

        # convert the data to ids
        self.ids = np.array(list(map(self.char2id.get, data)))

        self.alphabet_size = len(self.sorted_chars)        

    def encode(self, sequence):
        return np.array(list(map(self.char2id.get, sequence)))

    def decode(self, encoded_sequence):
        return np.array(list(map(self.id2char.get, encoded_sequence)))

    def create_minibatches(self, batch_size, sequence_length):
        batch_characters_size = batch_size * sequence_length
        num_batches = int((len(self.ids) - 1) / batch_characters_size)
        self.size = num_batches * batch_size * sequence_length

        input_output_pairs = np.array(list(zip(self.ids[:-1], self.ids[1:])))
         
#        print('using {} out of {} characters'.format(num_batches * batch_size * sequence_length, len(input_output_pairs)))
        input_output_pairs = input_output_pairs[:num_batches * batch_size * sequence_length]
        input_output_pairs = np.reshape(input_output_pairs, (batch_size, -1, 2))

        output = np.zeros_like(input_output_pairs.reshape((batch_size, -1, sequence_length, 2)))
        for index, batch_elem in enumerate(input_output_pairs):
            output[index] = np.split(batch_elem, output.shape[1])

        return output.transpose((1, 0, 2, 3))

    def to_one_hot(self, ids):
        one_hot = np.zeros(ids.shape + (self.alphabet_size,))
        for i in range(ids.shape[0]):
            for j in range (ids.shape[1]):
                one_hot[i, j, ids[i, j]] = 1
        return one_hot

def minibatches_iterator(minibatches):
    while True:
        minibatches, backup = tee(minibatches)
        new_epoch = True
        for minibatch in minibatches:
            x, y = np.transpose(minibatch, (2, 0, 1))
            yield new_epoch, x, y
            new_epoch = False
        minibatches = backup
        minibatches, backup = tee(minibatches)
