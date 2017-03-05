import numpy as np


def generate_batch(batch_size, seq_len, input_dim, dup_factor=3):

    target = np.random.randint(input_dim, size=(batch_size, seq_len))

    batch_output = np.zeros((batch_size, seq_len, input_dim))
    trans_input = np.zeros((seq_len*dup_factor, batch_size, input_dim))

    for i in range(batch_size):
        batch_output[i] = np.eye(input_dim)[target[i]]

    for i in range(seq_len):
        for j in range(dup_factor):
            trans_input[i*dup_factor+j] = batch_output[:, i, :]

    batch_input = np.transpose(trans_input, (1, 0, 2))

    return batch_input.astype(float), batch_output.astype(float)
