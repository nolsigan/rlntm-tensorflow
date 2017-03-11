import random
import numpy as np


class DataGenerator:

    def __init__(self, max_length, batch_size, num_symbols):
        self.max_length = max_length
        self.batch_size = batch_size
        self.num_symbols = num_symbols

    def __iter__(self):

        while True:

            batch = np.zeros((self.batch_size, self.max_length, self.num_symbols))
            for i in range(0, self.batch_size):
                for j in range(0, self.max_length):
                    k = random.randrange(0, self.num_symbols)
                    batch[i][j][k] = 1

            yield batch
