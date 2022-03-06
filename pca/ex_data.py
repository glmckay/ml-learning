import matplotlib.pyplot as plt
import pca
import scipy.io


data = scipy.io.loadmat("ex7data1.mat")["X"]

X = pca.normalize(data)

X_reduced = pca.reduce_data(X, projection_dimensions=1)

plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='red', alpha=0.5)
for i in range(data.shape[0]):
    a = X[i, :]
    b = X_reduced[i, :]
    plt.plot([a[0], b[0]], [a[1], b[1]], linestyle="--", color="black")
plt.show()
