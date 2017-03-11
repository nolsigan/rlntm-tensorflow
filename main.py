import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from attr_dict import AttrDict
from training import Training


def get_params():
    checkpoint_dir = '/Users/Nolsigan/PycharmProjects/rlntm-tensorflow/checkpoints'
    max_length = 6
    rnn_cell = rnn.BasicLSTMCell
    rnn_hidden = 128
    learning_rate = 0.003
    optimizer = tf.train.AdamOptimizer()
    gradient_clipping = 5
    batch_size = 100
    epochs = 3
    epoch_size = 100
    num_symbols = 10
    dup_factor = 2
    mem_dim = 128
    mem_move_table = [-1, 0, 1]
    in_move_table = [-1, 0, 1]
    out_move_table = [0, 1]
    return AttrDict(**locals())


Training(get_params())()
