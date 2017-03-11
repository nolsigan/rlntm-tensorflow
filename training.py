import tensorflow as tf
import os
import re
import numpy as np
from rlntm import RLNTM
from data_generator import DataGenerator
from utils import *


class Training:

    @overwrite_graph
    def __init__(self, params):
        self.params = params

        self.batches = DataGenerator(self.params.max_length, self.params.batch_size, self.params.num_symbols)
        self.input = tf.placeholder(tf.float32,
                                    [None, self.params.max_length, self.params.num_symbols], name='input')
        self.target = tf.placeholder(tf.float32,
                                     [None, self.params.max_length, self.params.num_symbols], name='target')
        self.model = RLNTM(self.params, self.input, self.target)
        self._init_or_load_session()

    def __call__(self):
        print('Start training!')

        self.logprobs = []
        batches = iter(self.batches)

        for epoch in range(self.epoch, self.params.epochs + 1):

            self.epoch = epoch
            for _ in range(self.params.epoch_size):
                self._optimization(next(batches))

            self._evaluation()

        writer = tf.summary.FileWriter('./my_graph', self.sess.graph)
        writer.close()

        return np.array(self.logprobs)

    def _optimization(self, batch):
        logprob, error = self.sess.run((self.model.logprob, self.model.error),
                                       {self.input: batch, self.target: batch})

        print(error)

        if np.isnan(logprob):
            raise Exception('training diverged')
        self.logprobs.append(logprob)

    def _evaluation(self):
        # self.saver.save(self.sess, os.path.join(
        #     self.params.checkpoint_dir, 'model'), self.epoch)
        perplexity = 2 ** -(sum(self.logprobs[-self.params.epoch_size:]) / self.params.epoch_size)

        print('Epoch {:2d} perplexity {:5.1f}'.format(self.epoch, perplexity))

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
