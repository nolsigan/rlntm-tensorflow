import tensorflow as tf
from data_generator import DuplicateData
from tape import Tape
from rlntm import RLNTM
from utils import *
import numpy as np


class Testing:

    @overwrite_graph
    def __init__(self, params):
        self.params = params

        self.total_length = self.params.max_length * self.params.dup_factor

        self.sequences = DuplicateData(self.params.max_length, 1,
                                       self.params.num_symbols, self.params.dup_factor, 3)

        self.state = tf.placeholder(tf.float32,
                                    [2, 1, self.params.rnn_hidden], name='state')
        self.input = tf.placeholder(tf.float32,
                                    [1, 1, self.params.num_symbols + self.params.rnn_hidden + 3], name='input')
        self.target = tf.placeholder(tf.float32,
                                     [1, 1, self.params.num_symbols], name='target')

        # generate dummy placeholders
        self.in_move = tf.placeholder(tf.float32,
                                      [1, 1, self.params.in_move_table.__len__()], name='in_move')
        self.out_move = tf.placeholder(tf.float32,
                                       [1, 1, self.params.out_move_table.__len__()], name='out_move')
        self.mem_move = tf.placeholder(tf.float32,
                                       [1, 1, self.params.mem_move_table.__len__()], name='mem_move')
        self.out_mask = tf.placeholder(tf.float32,
                                       [1, 1], name='out_mask')

        self.model = RLNTM(self.params, self.input, self.state, self.target,
                           self.in_move, self.out_move, self.mem_move, self.out_mask)

        self.sess = tf.Session()
        checkpoint = tf.train.get_checkpoint_state(self.params.checkpoint_dir)
        if checkpoint and checkpoint.model_checkpoint_path:
            tf.train.Saver().restore(
                self.sess, checkpoint.model_checkpoint_path)
        else:
            print('Checkpoint for testing not found')

    def __call__(self):
        print('Start testing!')

        test_sequences = iter(self.sequences)
        sequence = next(test_sequences)

        # create tapes
        input_tape = Tape(1, self.total_length, self.params.num_symbols, self.params.in_move_table,
                          initial=sequence)
        mem_tape = Tape(1, self.total_length, self.params.rnn_hidden, self.params.mem_move_table)
        output_tape = Tape(1, self.params.max_length, self.params.num_symbols, self.params.out_move_table,
                           initial=np.ones((1, self.params.max_length, self.params.num_symbols)) * -1)

        step = np.zeros((1, 1, self.params.num_symbols + self.params.rnn_hidden + 3))
        target = np.zeros((1, 1, self.params.num_symbols))
        state = np.zeros((2, 1, self.params.rnn_hidden))

        last_in_moves = np.ones((1, 1))
        last_mem_moves = np.ones((1, 1))
        last_out_moves = np.ones((1, 1))

        input_tape.print_tape()

        for i in range(sequence.shape[1]):
            print("=======================")
            print("step : ", i)
            # read input, memory from tape
            step_input = input_tape.read_tape()
            step_memory = mem_tape.read_tape()
            last_moves = np.concatenate((last_in_moves, last_mem_moves, last_out_moves), axis=1)
            step_concat = np.concatenate((step_input, step_memory, last_moves), axis=1)
            step[:, 0, :] = step_concat
            target[:, 0, :] = sequence[:, i, :]

            prediction, moves, hidden, state_tuple = self.sess.run([self.model.prediction, self.model.moves,
                                                                    self.model.hidden, self.model.state],
                                                                   {self.input: step, self.state: state,
                                                                    self.target: target})

            # update state
            state[0] = state_tuple[0]
            state[1] = state_tuple[1]

            # sample moves from logits
            # in_sample = sample(moves[0][:, 0, :])
            # mem_sample = sample(moves[1][:, 0, :])
            # out_sample = sample(moves[2][:, 0, :])
            in_sample = np.argmax(moves[0][:, 0, :], axis=1)
            mem_sample = np.argmax(moves[1][:, 0, :], axis=1)
            out_sample = np.argmax(moves[2][:, 0, :], axis=1)

            in_move = last_in_moves[:, 0] = input_tape.index_to_moves(in_sample)
            mem_move = last_mem_moves[:, 0] = mem_tape.index_to_moves(mem_sample)
            out_move = last_out_moves[:, 0] = output_tape.index_to_moves(out_sample)

            # if out_move, write to output_tape
            output_tape.write_tape(prediction[:, 0, :], out_move)

            # write memory
            mem_tape.write_tape(hidden[:, 0, :])

            # move ptrs
            input_tape.move_ptr(in_move)
            mem_tape.move_ptr(mem_move)
            output_tape.move_ptr(out_move)

            print("in_logits :", moves[0], ", in_move :", in_move, ", in_pos :", input_tape.get_ptr())
            print("mem_logits :", moves[1], ", mem_move :", mem_move, ", mem_pos :", mem_tape.get_ptr())
            print("out_logits :", moves[2], ", out_move :", out_move, ", out_pos :", output_tape.get_ptr())

            if out_move[0] == 1:
                print("prediction")
                print(prediction[:, 0, :])
                print("==> ", np.argmax(prediction[:, 0, :], axis=1))

            print("=======================")

        output_tape.print_max_indexes()
