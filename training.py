import tensorflow as tf
import os
import re
import numpy as np
from rlntm import RLNTM
from data_generator import DuplicateData
from tape import Tape
from utils import *


class Training:

    @overwrite_graph
    def __init__(self, params):
        self.params = params

        self.batches = DuplicateData(self.params.max_length, self.params.batch_size,
                                     self.params.num_symbols, self.params.dup_factor)

        total_length = self.params.max_length * self.params.dup_factor

        self.state = tf.placeholder(tf.float32,
                                    [2, None, self.params.rnn_hidden], name='state')
        self.input = tf.placeholder(tf.float32,
                                    [None, total_length, self.params.num_symbols + self.params.rnn_hidden + 3], name='input')
        self.target = tf.placeholder(tf.float32,
                                     [None, total_length, self.params.num_symbols], name='target')

        # fake labels for training RL
        self.in_move = tf.placeholder(tf.float32,
                                      [None, total_length, self.params.in_move_table.__len__()], name='in_move')
        self.out_move = tf.placeholder(tf.float32,
                                       [None, total_length, self.params.out_move_table.__len__()], name='out_move')
        self.mem_move = tf.placeholder(tf.float32,
                                       [None, total_length, self.params.mem_move_table.__len__()], name='mem_move')

        # mask for calculating gain
        self.out_mask = tf.placeholder(tf.float32,
                                       [None, total_length], name='out_mask')

        # initial input to make initial state
        self.init_input = tf.placeholder(tf.float32,
                                         [None, total_length, self.params.num_symbols], name='init_input')
        self.init_length = tf.placeholder(tf.float32,
                                          [None], name='init_length')

        self.model = RLNTM(self.params, self.input, self.state, self.target,
                           self.in_move, self.out_move, self.mem_move,
                           self.out_mask, self.init_input, self.init_length)
        self._init_or_load_session()

    def __call__(self):
        print('Start training!')

        self.costs = []
        batches = iter(self.batches)

        for epoch in range(self.epoch, self.params.epochs + 1):

            self.epoch = epoch
            for _ in range(self.params.epoch_size):
                self._optimization(next(batches))

            self._evaluation()

        writer = tf.summary.FileWriter('./my_graph', self.sess.graph)
        writer.close()

        return np.array(self.costs)

    def _optimization(self, batch):

        # create tapes
        input_tape = Tape(batch.shape[0], batch.shape[1], batch.shape[2], self.params.in_move_table,
                          initial=batch)
        mem_tape = Tape(batch.shape[0], batch.shape[1], self.params.rnn_hidden, self.params.mem_move_table)
        output_tape = Tape(batch.shape[0], batch.shape[1] // self.params.dup_factor, batch.shape[2],
                           self.params.out_move_table)

        # generate fake moves
        in_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.in_move_table.__len__()))
        mem_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.mem_move_table.__len__()))
        out_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.out_move_table.__len__()))

        in_moves[:, :, 2] = np.ones((batch.shape[0], batch.shape[1]))
        mem_moves[:, :, 1] = np.ones((batch.shape[0], batch.shape[1]))
        out_moves[:, :, 0] = np.ones((batch.shape[0], batch.shape[1]))
        # mem_moves[:, self.params.dup_factor-1::self.params.dup_factor, 1] \
        out_moves[:, 0::self.params.dup_factor, 0] \
            = np.zeros((batch.shape[0], batch.shape[1] // self.params.dup_factor))
        # mem_moves[:, self.params.dup_factor-1::self.params.dup_factor, 2] \
        out_moves[:, 0::self.params.dup_factor, 1] \
            = np.ones((batch.shape[0], batch.shape[1] // self.params.dup_factor))

        last_in_moves = np.ones((batch.shape[0], 1))
        last_mem_moves = np.ones((batch.shape[0], 1))
        last_out_moves = np.ones((batch.shape[0], 1))

        # output mask
        out_mask = np.zeros((batch.shape[0], batch.shape[1]))
        out_mask[:, 0::self.params.dup_factor] \
            = np.ones((batch.shape[0], batch.shape[1] // self.params.dup_factor))

        input_track = step = np.zeros((batch.shape[0], batch.shape[1], batch.shape[2] + self.params.rnn_hidden + 3))
        state = np.zeros((2, batch.shape[0], self.params.rnn_hidden))

        # go through input in reverse
        init_length = np.ones((batch.shape[0])) * batch.shape[1]
        state_tuple = self.sess.run(self.model.init_state,
                                    {self.state: state,
                                     self.init_input: batch[:, ::-1, :],
                                     self.init_length: init_length})

        state[0] = state_tuple[0]
        state[1] = state_tuple[1]

        for i in range(batch.shape[1]):
            # read from input_tape, memory_tape
            step_input = input_tape.read_tape()
            step_memory = mem_tape.read_tape()
            last_moves = np.concatenate((last_in_moves, last_mem_moves, last_out_moves), axis=1)
            step_concat = np.concatenate((step_input, step_memory, last_moves), axis=1)
            step[:, 0, :] = step_concat
            input_track[:, i, :] = step_concat

            moves, hidden, state_tuple = self.sess.run([self.model.moves, self.model.hidden, self.model.state],
                                                       {self.state: state,
                                                        self.input: step, self.target: batch,
                                                        self.in_move: in_moves, self.mem_move: mem_moves, self.out_move: out_moves,
                                                        self.out_mask: out_mask, self.init_input: batch})

            # update state
            state[0] = state_tuple[0]
            state[1] = state_tuple[1]

            # sample from logits
            #in_move = sample(moves[0])
            #mem_move = sample(moves[1])
            #out_move = sample(moves[2])

            # -- use fake moves for only-supervised-learning
            in_move_logits = in_moves[:, i, :].astype(np.int32)
            mem_move_logits = mem_moves[:, i, :].astype(np.int32)
            out_move_logits = out_moves[:, i, :].astype(np.int32)

            # write to memory
            mem_tape.write_tape(hidden[:, 0, :])

            # move ptrs
            in_move = last_in_moves[:, 0] = input_tape.index_to_moves(np.argmax(in_move_logits, axis=1))
            mem_move = last_mem_moves[:, 0] = mem_tape.index_to_moves(np.argmax(mem_move_logits, axis=1))
            out_move = last_out_moves[:, 0] = output_tape.index_to_moves(np.argmax(out_move_logits, axis=1))

            input_tape.move_ptr(in_move)
            mem_tape.move_ptr(mem_move)
            output_tape.move_ptr(out_move)

            # calculate gain

        state = np.zeros((2, batch.shape[0], self.params.rnn_hidden))
        cost, _ = self.sess.run((self.model.cost, self.model.optimize),
                                {self.state: state,
                                self.input: input_track, self.target: batch,
                                self.in_move: in_moves, self.mem_move: mem_moves, self.out_move: out_moves,
                                self.out_mask: out_mask})

        print(cost)
        self.costs.append(cost)

    def _evaluation(self):
        self.saver.save(self.sess, os.path.join(
            self.params.checkpoint_dir, 'model'), self.epoch)

        print('Epoch {:2d} cost {:1.10f}'.format(self.epoch, self.costs[-1]))

    def _init_or_load_session(self):
        self.sess = tf.Session()
        self.saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(self.params.checkpoint_dir)

        if checkpoint and checkpoint.model_checkpoint_path:
            path = checkpoint.model_checkpoint_path
            print('Load checkpoint', path)
            self.saver.restore(self.sess, path)
            self.epoch = int(re.search(r'-(\d+)$', path).group(1)) + 1
        else:
            ensure_directory(self.params.checkpoint_dir)
            print('Randomly initialize variables')
            self.sess.run(tf.global_variables_initializer())
            self.epoch = 1
