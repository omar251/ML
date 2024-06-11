import tensorflow as tf
# from tensorflow.keras.datasets import mnist
from tensorflow.keras import datasets, layers, models

# # Load the data and split it into train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# print('Training samples: {}'.format(len(x_train)))

# Load MNIST dataset from keras
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

# Reshape and normalize the images.
train_images = train_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Define the model architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1))) # 2D convolutional layer
model.add(layers.MaxPooling2D((2, 2)))                                              # Max pooling layer
model.add(layers.Conv2D(64, (3, 3), activation='relu'))                           # Another 2D convolutional layer
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())                                                         # Flatten the tensor output from the previous layer
model.add(layers.Dense(64, activation='relu'))                                     # Fully connected (dense) layer
model.add(layers.Dense(10))                                                          # Output layer

model = tf.keras.models.load_model('mnist_model.h5')

# Compile and train the model
model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=1, 
                    validation_data=(test_images, test_labels))

# Assuming that your model is named 'model'
model.save('mnist_model.h5')

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

import matplotlib.pyplot as plt

# Preview the first 25 test images
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.imshow(test_images[i].reshape((28, 28)), cmap='gray')
    plt.axis('off')
plt.show()