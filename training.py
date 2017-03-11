import tensorflow as tf
import os
import re
import numpy as np
from rlntm import RLNTM
from data_generator import DuplicateData
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
                                    [None, total_length, self.params.num_symbols], name='input')
        self.target = tf.placeholder(tf.float32,
                                     [None, total_length, self.params.num_symbols], name='target')
        self.in_move = tf.placeholder(tf.float32,
                                      [None, total_length, self.params.in_move_table.__len__()], name='in_move')
        self.out_move = tf.placeholder(tf.float32,
                                       [None, total_length, self.params.out_move_table.__len__()], name='out_move')
        self.mem_move = tf.placeholder(tf.float32,
                                       [None, total_length, self.params.mem_move_table.__len__()], name='mem_move')
        self.out_mask = tf.placeholder(tf.float32,
                                       [None, total_length], name='out_mask')
        self.model = RLNTM(self.params, self.input, self.target,
                           self.in_move, self.out_move, self.mem_move,
                           self.out_mask, self.state)
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

        # generate fake moves
        in_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.in_move_table.__len__()))
        mem_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.mem_move_table.__len__()))
        out_moves = np.zeros((batch.shape[0], batch.shape[1], self.params.out_move_table.__len__()))

        in_moves[:, :, 2] = np.ones((batch.shape[0], batch.shape[1]))
        mem_moves[:, 2::3, 2] = np.ones((batch.shape[0], batch.shape[1] // 3))
        out_moves[:, 1::3, 1] = np.ones((batch.shape[0], batch.shape[1] // 3))

        pred_list = []
        state = np.zeros((2, batch.shape[0], self.params.rnn_hidden))
        for i in range(batch.shape[1]):
            step = np.zeros(batch.shape)
            step[:, 0, :] = batch[:, 0, :]

            prediction, state_tuple = self.sess.run([self.model.prediction, self.model.state],
                                              {self.state: state,
                                               self.input: step, self.target: step,
                                               self.in_move: in_moves, self.mem_move: mem_moves, self.out_move: out_moves,
                                               self.out_mask: np.sum(mem_moves, axis=2)})

            state[0] = state_tuple[0]
            state[1] = state_tuple[1]

            pred_list.append(prediction[:, 0, :])

        state = np.zeros((2, batch.shape[0], self.params.rnn_hidden))
        cost, _ = self.sess.run((self.model.cost, self.model.optimize),
                                 {self.state: state,
                                  self.input: batch, self.target: batch,
                                  self.in_move: in_moves, self.mem_move: mem_moves, self.out_move: out_moves,
                                  self.out_mask: np.sum(mem_moves, axis=2)})

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
