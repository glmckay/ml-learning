import numpy as np
from utils import ImageData
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf  # noqa: E402


class ExpandedFeatureImageData(ImageData):
    def generate_data(self, *args, **kwargs):
        def expanded_features(x, y):
            return (
                x,
                y,
                x ** 2,
                x * y,
                y ** 2,
                x ** 3,
                x ** 2 * y,
                x * y ** 2,
                y ** 3,
                # x ** 4,
                # x ** 3 * y,
                # x ** 2 * y ** 2,
                # x * y ** 3,
                # y ** 4,
                # x ** 5,
                # x ** 4 * y,
                # x ** 3 * y ** 2,
                # x ** 2 * y ** 3,
                # x * y ** 4,
                # y ** 5,
            )

        features, labels = super().generate_data(*args, **kwargs)
        expanded = np.array([expanded_features(x, y) for x, y in features])
        return expanded, labels

    def plot_data(self, features, labels):
        pos_arr = [(x, y) for x, y, *_ in features]
        super().plot_data(pos_arr, labels)


img_data = ExpandedFeatureImageData("heart.png")

training_data = img_data.generate_data(10000)
test_data = img_data.generate_data(3000)
# cross_validation_data = generate_data(200)

model = tf.keras.Sequential(
    [
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(20, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation=tf.nn.sigmoid),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.03),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    # loss=tf.keras.losses.MeanSquaredError(),
    metrics=["accuracy"],
)

model.fit(*training_data, batch_size=10, epochs=10)

predicted_labels = model.predict(test_data[0])

img_data.plot_data(test_data[0], predicted_labels)
