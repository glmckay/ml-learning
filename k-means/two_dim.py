import numpy as np
import random
import scipy.io
from kmeans import run_k_means


def random_point():
    return (random.random() * 8, random.random() * 6)


# load data
data = scipy.io.loadmat("ex7data2.mat")["X"]

k = 3
initial_centroids = np.array([random_point() for i in range(k)])

run_k_means(
    data, k=k, iterations=10, initial_centroids=initial_centroids, show_plot=True
)
