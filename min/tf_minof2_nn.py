import tensorflow as tf
import numpy as np

# Define the training data
X = np.array([[53, 42], [67, 2], [81, 52], [16, 86], [84, 1], [64, 35], [93, 57], [10, 61], [63, 98], [72, 74]])
Y = np.array([[1], [1], [1], [0], [1], [1], [1], [0], [0], [0]])

# Create the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(2, activation='sigmoid', input_shape=(2,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# model = tf.keras.models.load_model('minof2_model.h5')
# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
# Train the model
model.fit(X, Y, epochs=1000)

# Save the model to a file
model.save('minof2_model.h5')
# Test the model
output = model.predict(X)
for i in range(len(X)):
    print(f"Input: {X[i]}, Output: {round(output[i][0])},Y: {X[i][round(output[i][0])]}, Expected Output: {output[i]}")

# #generate test data
# import random
# f = lambda : random.randint(0,100)
# g = lambda x: 1 if x[0] > x[1] else 0
# input = [[f(),f()] for _ in range(10)]
# output = [[g(i)] for i in input]
# print(input,output)


