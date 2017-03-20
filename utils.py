import functools
import tensorflow as tf
import numpy as np
import errno
import os


def lazy_property(function):
    attribute = '_' + function.__name__

    @property
    @functools.wraps(function)
    def wrapper(self):
        if not hasattr(self, attribute):
            setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return wrapper


def overwrite_graph(function):
    @functools.wraps(function)
    def wrapper(*args, **kwargs):
        with tf.Graph().as_default():
            return function(*args, **kwargs)

    return wrapper


def ensure_directory(directory):

    directory = os.path.expanduser(directory)
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def sample(logits):

    cumsum = np.cumsum(logits, axis=1)
    sample_val = np.random.rand(logits.shape[0])

    ptrs = np.zeros((logits.shape[0]), dtype=np.int32)

    for batch in range(logits.shape[0]):
        sample_ptr = 0
        for i in range(cumsum.shape[1]):
            if sample_val[batch] <= cumsum[batch][i]:
                break
            else:
                sample_ptr += 1

        ptrs[batch] = sample_ptr

    return ptrs
