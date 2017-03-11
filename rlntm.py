import tensorflow as tf
from utils import *


class RLNTM:

    def __init__(self, params, input_, target, initial=None):
        self.params = params
        self.input = input_
        self.target = target
        self.initial = initial

        self.prediction
        self.state
        self.cost
        self.error
        self.logprob
        self.optimize

    @lazy_property
    def mask(self):
        with tf.name_scope("mask"):
            return tf.reduce_max(tf.abs(self.target), reduction_indices=2)

    @lazy_property
    def length(self):
        with tf.name_scope("length"):
            return tf.reduce_sum(self.mask, reduction_indices=1)

    @lazy_property
    def prediction(self):
        with tf.name_scope("prediction"):
            prediction, _ = self.forward
            return prediction

    @lazy_property
    def state(self):
        with tf.name_scope("state"):
            _, state = self.forward

    @lazy_property
    def forward(self):
        with tf.name_scope("forward"):
            cell = self.params.rnn_cell(self.params.rnn_hidden)
            hidden, state = tf.nn.dynamic_rnn(
                inputs=self.input,
                cell=cell,
                dtype=tf.float32,
                initial_state=self.initial,
                sequence_length=self.length)

            num_symbols = int(self.target.get_shape()[2])
            max_length = int(self.target.get_shape()[1])

            weight = tf.Variable(tf.truncated_normal(
                [self.params.rnn_hidden, num_symbols], stddev=0.01))
            bias = tf.Variable(tf.constant(0.1, shape=[num_symbols]))

            output = tf.reshape(hidden, [-1, self.params.rnn_hidden])
            prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
            prediction = tf.reshape(prediction, [-1, max_length, num_symbols])

            return prediction, state

    @lazy_property
    def cost(self):
        with tf.name_scope("cost"):
            prediction = tf.clip_by_value(self.prediction, 1e-10, 1.0)
            cost = self.target * tf.log(prediction)
            cost = -tf.reduce_sum(cost, reduction_indices=2)
            return self._average(cost)

    @lazy_property
    def error(self):
        with tf.name_scope("error"):
            error = tf.not_equal(tf.argmax(self.prediction, 2), tf.argmax(self.target, 2))
            error = tf.cast(error, tf.float32)
            return self._average(error)

    @lazy_property
    def logprob(self):
        with tf.name_scope("logprob"):
            logprob = tf.multiply(self.prediction, self.target)
            logprob = tf.reduce_max(logprob, reduction_indices=2)
            logprob = tf.log(tf.clip_by_value(logprob, 1e-10, 1.0)) / tf.log(2.0)
            return self._average(logprob)

    @lazy_property
    def optimize(self):
        with tf.name_scope("optimize"):

            gradient = self.params.optimizer.compute_gradients(self.cost)
            if self.params.gradient_clipping:
                limit = self.params.gradient_clipping
                gradient = [
                    (tf.clip_by_value(g, -limit, limit), v)
                    if g is not None else (None, v)
                    for g, v in gradient
                ]
            optimize = self.params.optimizer.apply_gradients(gradient)
            return optimize

    def _average(self, data):
        with tf.name_scope("average"):
            data *= self.mask
            length = tf.reduce_sum(self.length, 0)
            data = tf.reduce_sum(data, reduction_indices=1) / length
            data = tf.reduce_mean(data)
            return data



