import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Define the XOR gate input and output
xor_input = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype="float32")
xor_output = np.array([[0], [1], [1], [0]], dtype="float32")

# Create the Sequential model
model = Sequential()

# Add layers to the model
model.add(Dense(2, input_dim=2, activation='sigmoid'))  # First hidden layer with 2 neurons
model.add(Dense(1, activation='sigmoid'))  # Output layer for the XOR problem

# Compile the model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.01))

# Train the model
model.fit(xor_input, xor_output, epochs=5000, verbose=2)

# Evaluate the model
loss = model.evaluate(xor_input, xor_output, verbose=0)
print("Final loss: {:.4f}".format(loss))

# Make predictions
predictions = model.predict(xor_input)
print("Predictions:\n{}".format(predictions))

# Retrieve the weights and biases
weights_layer1, biases_layer1 = model.layers[0].get_weights()
weights_layer2, biases_layer2 = model.layers[1].get_weights()

print("Weights for layer 1: {}".format(weights_layer1))
print("Biases for layer 1: {}".format(biases_layer1))
print("Weights for layer 2: {}".format(weights_layer2))
print("Biases for layer 2: {}".format(biases_layer2))

