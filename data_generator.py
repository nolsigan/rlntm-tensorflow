import random
import numpy as np


class DuplicateData:

    def __init__(self, max_length, batch_size, num_symbols, dup_factor, min_length=1):
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_symbols = num_symbols
        self.dup_factor = dup_factor
        self.min_length = min_length

    def __iter__(self):

        while True:

            batch = np.zeros((self.batch_size, self.max_length * self.dup_factor, self.num_symbols))
            for i in range(0, self.batch_size):
                length = random.randrange(self.min_length, self.max_length)
                for j in range(0, length):
                    k = random.randrange(0, self.num_symbols)
                    for dup in range(0, self.dup_factor):
                        batch[i][j*self.dup_factor+dup][k] = 1

            yield batch
