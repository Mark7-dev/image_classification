import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models

#
(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images, testing_images = training_images / 255.0, testing_images / 255.0

#defining the class names
class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()

#Code below is all code to train the model.


model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2,2)))

# removes features of an image to its essential components

model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# convolutional layer filters for features in an image.
# For example, it will recognise that a horse has long legs
# whereas a cat has pointy ears
# The maxpooling then removes features of an image to its essential parts


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_images, training_labels)

model.fit(training_images, training_labels, epochs=20, validation_data=(testing_images, testing_labels))

loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('model.classifier.model')

#code below is to test the image recognition

# model = models.load_model('/Users/yourpath/PycharmProjects/image_classification/model.classifier.keras')
#
#
# img = cv.imread('deer.jpg')
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
#
# plt.imshow(img, cmap=plt.cm.binary)
# prediction = model.predict(np.array([img]) / 255)
# index = np.argmax(prediction)
# print(f'Prediction is : {class_names[index]}')