import matplotlib.pyplot as plt
import math
import numpy as np
import pca
import scipy.io


def plot_faces(X):
    """helper to show grid of faces"""

    num_faces, face_size = X.shape
    face_width = round(math.sqrt(face_size))
    face_height = face_size // face_width

    # number of rows and columns of faces
    rows = int(math.floor(math.sqrt(num_faces)))
    cols = int(math.ceil(num_faces / rows))

    pad = 1  # grid padding

    plt.set_cmap("gray")
    # black image (apparently 1 is black...)
    buffer = -np.ones(
        (pad + rows * (face_height + pad), pad + cols * (face_width + pad))
    )

    # Copy each example into a patch on the display array
    for n, Y in enumerate(X):
        i, j = divmod(n, cols)

        # Reshape and normalize brightness
        # Fortran order means columns are contiguous (I think matlab does the same)
        face = 1 / max(abs(Y)) * np.reshape(Y, (face_height, face_width), order="F")

        # indices where we are copying into the buffer
        i_min = pad + (n // cols) * (face_height + pad)
        j_min = pad + (n % cols) * (face_width + pad)
        i_max = i_min + face_height
        j_max = j_min + face_width

        buffer[i_min:i_max, j_min:j_max] = face

    plt.imshow(buffer, vmin=-1, vmax=1)


data = scipy.io.loadmat("ex7faces.mat")["X"]

X = pca.normalize(data)

# view some eigenvectors
# show_faces(U[:, :36].T)
# plt.show()

X_reduced = pca.reduce_data(X, projection_dimensions=1000)

# show 100 faces and the corresponding projec
plt.subplot(1, 2, 1)
plot_faces(X[:100, :])
plt.subplot(1, 2, 2)
plot_faces(X_reduced[:100, :])
plt.show()
