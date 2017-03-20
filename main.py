import sys
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from attr_dict import AttrDict
from training import Training
from testing import Testing


def get_params():
    checkpoint_dir = '/Users/Nolsigan/PycharmProjects/rlntm-tensorflow/checkpoints'
    max_length = 6
    rnn_cell = rnn.BasicLSTMCell
    rnn_hidden = 128
    learning_rate = 0.003
    optimizer = tf.train.AdamOptimizer()
    gradient_clipping = 5
    batch_size = 100
    epochs = 30
    epoch_size = 100
    num_symbols = 10
    dup_factor = 2
    mem_dim = 128
    mem_move_table = [-1, 0, 1]
    in_move_table = [-1, 0, 1]
    out_move_table = [0, 1]
    return AttrDict(**locals())


mode = sys.argv[1]

if mode == '--train':
    Training(get_params())()
elif mode == '--test':
    Testing(get_params())()
else:
    print('no mode specified, please use --train or --test as first argument')
