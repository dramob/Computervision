import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Sequential

# Define a function to create the model with dropout regularization
def define_model_with_dropout(dropout_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Dropout(dropout_rate),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the dataset and preprocess it
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Dropout rates to evaluate
dropout_rates = [0.1, 0.2, 0.3, 0.4, 0.5]

# Store the results
results = []

for dropout_rate in dropout_rates:
    # Start the clock
    start_time = time.time()

    # Create and train the model
    model = define_model_with_dropout(dropout_rate)
    model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=2)

    # Evaluate the model
    _, accuracy = model.evaluate(testX, testY, verbose=0)

    # Stop the clock
    end_time = time.time()

    # Print the results
    print(f'Dropout Rate: {dropout_rate}, Test Accuracy: {accuracy:.3f}, Time: {end_time - start_time:.2f}s')

    # Append to results
    results.append((dropout_rate, accuracy, end_time - start_time))

# Display a summary of results
print("\nSummary of Results:")
for dropout_rate, accuracy, time_taken in results:
    print(f'Dropout Rate: {dropout_rate}, Accuracy: {accuracy:.3f}, Training Time: {time_taken:.2f}s')
