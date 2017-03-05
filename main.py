import tensorflow as tf
import numpy as np
from model import Model
from batch_generator import generate_batch


BATCH_SIZE = 30
DUP_FACTOR = 3
NUM_SYMBOLS = 5
SEQ_LEN = 3
BATCH_ITER = 1000
MAX_STEPS = 15
MEM_DIM = 128

rlntm = Model(seq_len=SEQ_LEN, num_symbols=NUM_SYMBOLS, batch_size=BATCH_SIZE, dup_factor=DUP_FACTOR,
                  emb_dim=NUM_SYMBOLS, max_steps=MAX_STEPS, mem_dim=MEM_DIM)

with tf.Session() as sess:
    print("!!!!!!!!!!!!!!!!!!!")

    print("~~~~~~~~~~~~~~~~~~~~~")
    init = tf.global_variables_initializer()
    sess.run(init)

    for i in range(BATCH_ITER):
        print("========================================")
        print("batch : " + i)
        batch_input, batch_output = generate_batch(BATCH_SIZE, SEQ_LEN, NUM_SYMBOLS, DUP_FACTOR)

        mem_tape = np.zeros([BATCH_SIZE, SEQ_LEN * DUP_FACTOR, MEM_DIM], dtype=np.float32).astype(float)
        print(mem_tape)

        rlntm.train(sess, feed_dict={rlntm.input_tape: batch_input, rlntm.output_tape: batch_output})
