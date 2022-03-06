import cv2
import matplotlib.pyplot as plt
import numpy as np
import random
from kmeans import run_k_means, closest_centroids


def random_colour():
    return np.array([random.random() for _ in range(3)])


# load data (cv2 gives colours as BGR and matplotlib expects RGB, convert to RGB)
img = cv2.imread("bird_small.png")[:, :, ::-1]
height, width, *_ = img.shape

# colour values between 0 and 1
data = (img / 255).reshape((height * width, 3))

k = 16
initial_centroids = np.array([random_colour() for _ in range(k)])

colours = run_k_means(
    data, k=k, iterations=10, initial_centroids=initial_centroids, show_plot=False
)

compressed = colours[closest_centroids(data, colours)].reshape((height, width, 3))

plt.imshow(compressed)
plt.show()
