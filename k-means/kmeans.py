import matplotlib.pyplot as plt
import matplotlib.animation as plt_animation
import numpy as np
import random
import scipy.spatial.distance


def init_centroids(ground_set, k: int):
    """Randomly pick k centroids"""
    n = ground_set.shape[0]
    assert n >= k
    return ground_set[random.sample(list(range(n)), k)]


def closest_centroids(elements, centroids):
    """Returns array of indices corresponding to closest centroids for each element"""
    return np.array(
        [scipy.spatial.distance.cdist([e], centroids).argmin() for e in elements]
    )


def centroid_means(elements, closest_centroid, k: int):
    """Computes mean of points closest to each centroid
    Assumes elements is numpy array"""

    # a centroid may not be closest to any elements
    default = np.zeros(elements.shape[1:])
    elements_per_centroid = (
        elements[np.nonzero(closest_centroid == i)] for i in range(k)
    )
    return np.array(
        [es.mean(axis=0) if es.size != 0 else default for es in elements_per_centroid]
    )


def plot_k_means_progress(elements, closest_centroid_history, centroid_history):
    k = centroid_history[-1].size

    fig = plt.figure()

    def show_frame(i):
        fig.clear()

        colours = closest_centroid_history[i] / (k + 1)
        plt.scatter(
            elements[:, 0],
            elements[:, 1],
            c=colours,
            cmap=plt.cm.get_cmap("rainbow"),
            edgecolors="black",
            alpha=0.3,
        )

        previous = None
        for centroids in centroid_history[:i]:
            plt.scatter(centroids[:, 0], centroids[:, 1], marker="x", s=60, c="black")
            if previous is not None:
                for c, p in zip(centroids, previous):
                    plt.plot([c[0], p[0]], [c[1], p[1]], c="black")
            previous = centroids

        plt.draw()

    anim = plt_animation.FuncAnimation(  # noqa: F841
        fig, show_frame, len(centroid_history), repeat=True, interval=400
    )
    plt.show()


def run_k_means(
    elements,
    k: int,
    iterations,
    initial_centroids=None,
    initial_centroid_ground_set=None,
    show_plot=False,
):
    if initial_centroids is None:
        if initial_centroid_ground_set is None:
            initial_centroid_ground_set = elements
        centroids = init_centroids(initial_centroid_ground_set, k)
    else:
        centroids = initial_centroids

    closest_history = []
    centroid_history = []

    for i in range(iterations):
        closest_indices = closest_centroids(elements, centroids)
        centroids = centroid_means(elements, closest_indices, k)
        if show_plot:
            closest_history.append(closest_indices)
            centroid_history.append(centroids)
    if show_plot:
        plot_k_means_progress(elements, closest_history, centroid_history)

    return centroids
