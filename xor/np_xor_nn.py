import numpy as np

# Define the number of inputs, hidden units, and outputs
n_inputs = 2
n_hidden = 2
n_outputs = 1

# Define the activation functions for the hidden and output layers
sigmoid = lambda x: 1 / (1 + np.exp(-x))
sigmoid_derivative = lambda x: x * (1 - x)

# Initialize the weights and biases for the layers
weights1 = np.random.rand(n_inputs, n_hidden)
weights2 = np.random.rand(n_hidden, n_outputs)
bias1 = np.zeros((1, n_hidden))
bias2 = np.zeros((1, n_outputs))

# Define the training data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the cost function
cost_function = lambda y, output: np.mean((y - output) ** 2)
# Set the learning rate
learning_rate = 0.5

# Train the network
for i in range(1000):
    # Forward pass
    hidden_layer = sigmoid(np.dot(X, weights1) + bias1)
    output_layer = sigmoid(np.dot(hidden_layer, weights2) + bias2)

    # Calculate the cost
    cost = cost_function(y, output_layer)

    # Print the cost at each iteration
    print(f"Iteration {i+1}, Cost: {cost:.4f}")

    # Backward pass
    output_delta = 2 * (y - output_layer) * sigmoid_derivative(output_layer)
    hidden_delta = output_delta.dot(weights2.T) * sigmoid_derivative(hidden_layer)

    # Update the weights and biases
    weights2 += hidden_layer.T.dot(output_delta) * learning_rate
    bias2 += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
    weights1 += X.T.dot(hidden_delta) * learning_rate
    bias1 += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate

# Save the weights and biases to a file
np.save('weights1.npy', weights1)
np.save('weights2.npy', weights2)
np.save('bias1.npy', bias1)
np.save('bias2.npy', bias2)

# Load the weights and biases from a file
loaded_weights1 = np.load('weights1.npy')
loaded_weights2 = np.load('weights2.npy')
loaded_bias1 = np.load('bias1.npy')
loaded_bias2 = np.load('bias2.npy')

# Test the network with loaded weights and biases
hidden_layer = sigmoid(np.dot(X, loaded_weights1) + loaded_bias1)
output_layer = sigmoid(np.dot(hidden_layer, loaded_weights2) + loaded_bias2)

print("Testing the network with loaded weights and biases:")
print("Input:", X)
print("Output:", output_layer)
