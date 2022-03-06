from sklearn import svm
from utils import ImageData

img_data = ImageData("star.png")

training_data = img_data.generate_data(3000)
test_data = img_data.generate_data(3000)
# cross_validation_data = generate_data(200)

classifier = svm.SVC(C=10000, kernel="rbf")

classifier.fit(*training_data)
print(f"accuracy after training: {classifier.score(*test_data)}")

predicted_labels = classifier.predict(test_data[0])

# plot_data(*training_data)

img_data.plot_data(test_data[0], predicted_labels)
