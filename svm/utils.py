import cv2
import itertools
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np


class ImageData:
    def __init__(self, image_path):
        self.img = cv2.imread(image_path)

    def generate_data(self, num_elements: int):
        def pixel_label(val):
            # label is black or non-black, so we just need to check oen colour channel
            return 0 if val[0] < 0.1 * 255 else 1

        height, width, *_ = self.img.shape
        assert num_elements <= height * width
        indices = list(map(tuple, itertools.product(range(height), range(width))))

        sampled = list(random.sample(indices, num_elements))
        pos_arr = np.array([(i / height, j / width) for i, j in sampled])
        label_arr = np.array(list(pixel_label(self.img[i, j]) for i, j in sampled))
        return pos_arr, label_arr

    def plot_data(self, pos_arr, label_arr):
        y = [1 - pos[0] for pos in pos_arr]
        x = [pos[1] for pos in pos_arr]

        plt.scatter(x, y, c=label_arr, cmap=matplotlib.cm.get_cmap("bwr"), alpha=0.3)
        plt.show()
