import tensorflow as tf
import tensorflow.contrib.rnn as rnn
import numpy as np


class Model(object):
    def __init__(self, num_symbols=10,
                 state_dim=128, batch_size=10, seq_len=3, mem_dim=128,
                 emb_dim=10, dup_factor=3,
                 max_steps=15):

        self.num_symbols = num_symbols
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.mem_dim = mem_dim
        self.emb_dim = emb_dim
        self.dup_factor = dup_factor
        self.max_steps = max_steps
        self.global_step = 1

        self.in_move_table = [-1, 0, 1]
        self.mem_move_table = [-1, 0, 1]
        self.out_move_table = [0, 1]

        self.is_training = tf.placeholder_with_default(
            tf.constant(False, dtype=tf.bool),
            shape=(), name='is_training')

        self._build_model()
        self._build_optim()

        self._build_steps()

    def _build_model(self):
        print("_build_model")
        self.input_tape = tf.placeholder(np.float32, [self.batch_size, self.seq_len * self.dup_factor, self.num_symbols],
                                         name="input_tape")
        self.output_tape = tf.placeholder(np.float32, [self.batch_size, self.seq_len, self.num_symbols],
                                          name="output_tape")
        self.mem_tape = np.zeros([self.batch_size, self.seq_len * self.dup_factor, self.mem_dim], dtype=np.float32)

        self.input_pos = np.zeros([self.batch_size], dtype=np.int32)
        self.output_pos = np.zeros([self.batch_size], dtype=np.int32)
        self.mem_pos = np.zeros([self.batch_size], dtype=np.int32)

        self.cell = rnn.BasicLSTMCell(self.state_dim)

        with tf.variable_scope("lstm", initializer=tf.random_normal_initializer(stddev=0.5)) as lstm_scope:
            self.state = self.cell.zero_state(self.batch_size, dtype=tf.float32)

        lstm_scope.reuse_variables()

        # self.last_moves = tf.zeros([self.batch_size, 3, self.emb_dim])

        self.in_dist_list = []
        self.mem_dist_list = []
        self.out_dist_list = []
        self.pred_dist_list = []

        self.sample_list = []

        self.total_gain = np.zeros([self.batch_size])
        self.gain_list = []

        for i in range(self.max_steps):

            reuse = (i != 0)

            # read from memory -> make the_input
            input_t = tf.concat([self.read_mem(), self.get_input()], 1)

            # LSTM Controller
            output_t = self._build_controller(input_t)

            # calc logits of in, mem, out moves
            self._build_move_dists(tf.scalar_mul(0.1, output_t), reuse=reuse)

            # calc logits of output prediction
            self._build_pred_dist(output_t, reuse=reuse)

            # RL sampling using in, mem, out move logits
            in_move = self.rl_sample(self.in_dist_list[i])
            mem_move = self.rl_sample(self.mem_dist_list[i])
            out_move = self.rl_sample(self.out_dist_list[i])

            self.sample_list[i] = np.array([in_move, mem_move, out_move], dtype=np.int32)

            # write to mem
            self.write_mem(output_t)

            # store gain value for RL feed back
            self.gain_list[i] = self.calc_gain(out_move)
            self.total_gain += self.gain_list[i]

            # move ptrs
            self.move_ptr(in_move, mem_move, out_move)

        # calc advantage for loss function
        self.advantage = tf.cumsum(self.gain_list, axis=0, reverse=True)

    def read_mem(self):

        cur_mem = None
        for i in range(self.batch_size):
            mem_val = tf.stack([self.mem_tape[i][self.mem_pos[i]]])
            if cur_mem is None:
                cur_mem = mem_val
            else:
                cur_mem = tf.concat([cur_mem, mem_val], 0)

        return cur_mem

    def write_mem(self, output_t):

        with tf.variable_scope("write_mem", reuse=True):
            write_weights = tf.get_variable("write_weights", [self.state_dim, self.mem_dim], tf.float32,
                                            tf.random_normal_initializer(stddev=0.5))
            write_biases = tf.get_variable("write_biases", [self.batch_size, self.mem_dim], tf.float32,
                                           tf.random_normal_initializer(stddev=0.5))

        new_mem = tf.nn.bias_add(tf.matmul(output_t, write_weights), write_biases)

        self.mem_tape[:, self.mem_pos, :] = new_mem

    def get_output(self):

        output = None
        for i in range(self.batch_size):

            cur_out = tf.stack([self.output_tape[i][self.output_pos[i]]])

            if output is None:
                output = cur_out
            else:
                output = tf.concat([output, cur_out], 0)

        return output

    def get_input(self):

        input_ = None
        for i in range(self.batch_size):
            cur_input = tf.stack([self.input_tape[i][self.input_pos[i]]])

            if input_ is None:
                input_ = cur_input
            else:
                input_ = tf.concat([input_, cur_input], 0)

        return input_

    def _build_controller(self, input_t):
        output, self.state = self.cell(input_t, self.state)
        return output

    def _build_move_dists(self, output_t, reuse=False):

        # in_dist
        if reuse:
            with tf.variable_scope("in_move", reuse=True):
                in_weights = tf.get_variable("in_weights")
                in_biases = tf.get_variable("in_biases")
        else:
            with tf.variable_scope("in_move"):
                in_weights = tf.get_variable("in_weights", [self.state_dim, len(self.in_move_table)], tf.float32,
                                             tf.random_normal_initializer(stddev=0.5))
                in_biases = tf.get_variable("in_biases", [self.batch_size, len(self.in_move_table)], tf.float32,
                                            tf.random_normal_initializer(stddev=0.5))

        in_dist = tf.add(tf.matmul(output_t, in_weights), in_biases)
        in_dist_sm = tf.nn.softmax(in_dist)
        in_dist_log = tf.log(in_dist_sm)

        # mem_dist
        if reuse:
            with tf.variable_scope("mem_move", reuse=True):
                mem_weights = tf.get_variable("mem_weights")
                mem_biases = tf.get_variable("mem_biases")
        else:
            with tf.variable_scope("mem_move"):
                mem_weights = tf.get_variable("mem_weights", [self.state_dim, len(self.mem_move_table)], tf.float32,
                                              tf.random_normal_initializer(stddev=0.5))
                mem_biases = tf.get_variable("mem_biases", [self.batch_size, len(self.mem_move_table)], tf.float32,
                                             tf.random_normal_initializer(stddev=0.5))

        mem_dist = tf.add(tf.matmul(output_t, mem_weights), mem_biases)
        mem_dist_sm = tf.nn.softmax(mem_dist)
        mem_dist_log = tf.log(mem_dist_sm)

        # out_dist
        if reuse:
            with tf.variable_scope("out_move", reuse=True):
                out_weights = tf.get_variable("out_weights")
                out_biases = tf.get_variable("out_biases")
        else:
            with tf.variable_scope("out_move"):
                out_weights = tf.get_variable("out_weights", [self.state_dim, len(self.out_move_table)], tf.float32,
                                              tf.random_normal_initializer(stddev=0.5))
                out_biases = tf.get_variable("out_biases", [self.batch_size, len(self.out_move_table)], tf.float32,
                                             tf.random_normal_initializer(stddev=0.5))

        out_dist = tf.add(tf.matmul(output_t, out_weights), out_biases)
        out_dist_sm = tf.nn.softmax(out_dist)
        out_dist_log = tf.log(out_dist_sm)

        self.in_dist_list.append(in_dist_log)
        self.mem_dist_list.append(mem_dist_log)
        self.out_dist_list.append(out_dist_log)

    def _build_pred_dist(self, output_t, reuse=None):

        if reuse:
            with tf.variable_scope("pred_move", reuse=True):
                pred_weights = tf.get_variable("pred_weights")
                pred_biases = tf.get_variable("pred_biases")
        else:
            with tf.variable_scope("pred_move"):
                pred_weights = tf.get_variable("pred_weights", [self.state_dim, self.num_symbols], tf.float32,
                                              tf.random_normal_initializer(stddev=0.5))
                pred_biases = tf.get_variable("pred_biases", [self.batch_size, self.num_symbols], tf.float32,
                                             tf.random_normal_initializer(stddev=0.5))

        pred_dist = tf.add(tf.matmul(output_t, pred_weights), pred_biases)
        pred_dist_sm = tf.nn.softmax(pred_dist)
        pred_dist_log = tf.log(pred_dist_sm)

        self.pred_dist_list.append(pred_dist_log)

    def rl_sample(self, dist):

        dist_sm = tf.exp(dist)
        dist_cumsum = tf.cumsum(dist_sm, axis=1)

        rnd = np.random.random_sample()

        indications = tf.Tensor.__le__(dist_cumsum, rnd)
        samples = tf.reduce_sum(tf.cast(indications, tf.int32), 1)

        return samples

    def calc_gain(self, out_move):

        target_t = self.get_output()
        mask_t = tf.ones([self.batch_size, self.num_symbols], tf.float32)
        true_mask_t = tf.scalar_mul(mask_t, out_move)
        unmasked_gain_t = tf.reduce_sum(self.pred_dist_list[-1] * target_t, 1)
        masked_gain_t = unmasked_gain_t * true_mask_t
        gain_t = masked_gain_t # add remaining

        return gain_t

    def move_ptr(self, in_move, mem_move, out_move):

        self.input_pos += self.in_move_table[in_move]
        self.mem_pos += self.mem_move_table[mem_move]
        self.output_pos += self.out_move_table[out_move]

    def _build_optim(self):

        with tf.variable_scope("optim"):

            log_prob = np.zeros([self.max_steps, self.batch_size], dtype=np.float32)
            for i in range(self.max_steps):
                log_prob[i] = tf.reduce_max(self.in_dist_list[i], reduction_indices=[1])

            cost = -self.total_gain - np.mean(np.sum(np.multiply(log_prob, self.advantage), axis=0))

            optimizer = tf.train.AdamOptimizer(learning_rate=0.03)

            self.loss = cost
            self.optim = optimizer.minimize(cost)

    def _build_steps(self):

        def train(sess, feed_dict={}):
            self.global_step += 1

            sess.run(self.optim, feed_dict=feed_dict)

            if self.global_step % 10 == 0:
                loss = sess.run(self.loss, feed_dict=feed_dict)
                print("loss : {:.6f}".format(loss))

        self.train = train
