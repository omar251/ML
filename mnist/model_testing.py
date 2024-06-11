
import tensorflow as tf
# from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models

# Load MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape and normalize the images.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

model = tf.keras.models.load_model('mnist_model.h5')


predictions = model.predict(test_images)
for i in range(10):
    pred_index = predictions[i].argmax()
    print("Image", i)
    print("Predicted class:", pred_index)
    print("Actual label:", test_labels[i])
    print()

import matplotlib.pyplot as plt

# Preview the first 25 test images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i].reshape((28, 28)), cmap='gray')
    plt.axis('off')
plt.show()
