import functools
import numpy as np
from sklearn import svm
from utils import ImageData

img_data = ImageData("alpha.png")

cross_validation_data = img_data.generate_data(600)
training_data = img_data.generate_data(1000)
test_data = img_data.generate_data(3000)  # big so that plot looks nice


def gaussian_kernel(x, y, sigma=0.025):
    def kern(x, y):
        return np.exp(-np.sum(np.power(x - y, 2)) / (2 * sigma ** 2))

    return np.array([[kern(xi, yj) for yj in y] for xi in x])


# scores = {}
# for i in range(10):
#     sigma = 0.025 + 0.005 * i
#     kernel = functools.partial(gaussian_kernel, sigma=sigma)
#     classifier = svm.SVC(C=100, kernel=kernel)

#     classifier.fit(*training_data)
#     score = classifier.score(*cross_validation_data)
#     print(f"{sigma=:<5}: accuracy{score}")
#     scores[sigma] = score

# best_sigma, _ = max(scores.items(), key=lambda kv: kv[1])

# print(f"best sigma is {best_sigma}")

kernel = functools.partial(gaussian_kernel, sigma=0.3)
classifier = svm.SVC(C=100, kernel=kernel)
classifier.fit(*training_data)

predicted_labels = classifier.predict(test_data[0])
img_data.plot_data(test_data[0], predicted_labels)
