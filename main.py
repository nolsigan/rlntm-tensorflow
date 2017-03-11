import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from attr_dict import AttrDict
from training import Training


def get_params():
    checkpoint_dir = '/Users/Nolsigan/PycharmProjects/rlntm-tensorflow/checkpoints'
    max_length = 10
    rnn_cell = rnn.BasicLSTMCell
    rnn_hidden = 128
    learning_rate = 0.002
    optimizer = tf.train.AdamOptimizer()
    gradient_clipping = 5
    batch_size = 100
    epochs = 3
    epoch_size = 200
    num_symbols = 10
    return AttrDict(**locals())


Training(get_params())()
