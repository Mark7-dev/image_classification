# Image Classifier Using CIFAR-10 Dataset

This project is a basic image classifier built using the CIFAR-10 dataset and implemented with TensorFlow's Keras API. The classifier uses a convolutional neural network (CNN) to categorize images into one of ten classes, including airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)

## Installation

To get started, you'll need to have Python installed along with the following libraries:

- TensorFlow
- NumPy
- OpenCV (cv2)
- Matplotlib

You can install these dependencies using `pip`:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

## Usage

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/image-classifier.git
   cd image-classifier
   ```

2. Run the script:

   ```bash
   python image_classifier.py
   ```

This will train the model and evaluate its performance on the CIFAR-10 test set.

## Model Architecture

The model consists of a series of convolutional and max-pooling layers followed by fully connected layers. The architecture is as follows:

- **Conv2D Layer**: 32 filters, kernel size 3x3, activation ReLU
- **MaxPooling2D Layer**: pool size 2x2
- **Conv2D Layer**: 64 filters, kernel size 3x3, activation ReLU
- **MaxPooling2D Layer**: pool size 2x2
- **Conv2D Layer**: 64 filters, kernel size 3x3, activation ReLU
- **Flatten Layer**
- **Dense Layer**: 64 units, activation ReLU
- **Dense Layer**: 10 units (for the 10 classes), activation Softmax

## Training

The model is trained using the CIFAR-10 training set. The images are normalised by scaling the pixel values to the range [0, 1]. The model is compiled with the Adam optimiser and trained for 10 epochs using the sparse categorical cross-entropy loss function.

To train the model, run:

```python
model.fit(training_images, training_labels, epochs=10, validation_data=(testing_images, testing_labels))
```

## Evaluation

After training, the model is evaluated on the test set to measure its performance:

```python
loss, accuracy = model.evaluate(testing_images, testing_labels)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")
```

## Saving and Loading the Model

The trained model can be saved to a file for later use:

```python
model.save('model.classifier.model')
```

To load the saved model:

```python
model = models.load_model('image_classifier.model')
```

Bibliography:

https://www.youtube.com/watch?v=t0EzVCvQjGE
