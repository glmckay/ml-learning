import gzip
import pickle
import random

import numpy as np

from itertools import zip_longest, repeat
from network import Network



def grouper(iterable, n, fillvalue=None):
    """Collect data into fixed-length chunks or blocks"""
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def load_data(path):
    def format_data(inputs, labels):
        return zip((np.reshape(i, (784, 1)) for i in inputs), labels)

    with gzip.open(path, "rb") as f:
        return (list(format_data(*data)) for data in pickle.load(f, encoding="latin1"))


def indicator_vector(n):
    v = np.zeros((10, 1))
    v[n] = 1
    return v


def prep_training(example):
    inputs, label = example
    return (inputs, indicator_vector(label))


training, validation, testing = load_data("mnist.pkl.gz")

# training = training * 3

random.shuffle(training)

training = map(prep_training, training)


N = Network(784, 10, [16, 16])


correct = N.evaluate(testing)
print(f"{correct} out of {10000}")

n = 0
for examples in grouper(training, 100):
    N.train(examples, 2)

    n += 1
    if n >= 50:
        correct = N.evaluate(testing)
        print(f"{correct} out of {10000}")
        n = 0

# correct = N.evaluate(testing)
# print(f"{correct} out of {10000}")
