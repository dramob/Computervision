import numpy as np
from keras.datasets import fashion_mnist
from keras.utils  import to_categorical
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD

def create_model(input_shape=(28, 28, 1), num_classes=10, num_filters=32, kernel_size=(3, 3), dense_neurons=32, learning_rate=0.01):
    model = Sequential([
        Conv2D(num_filters, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(dense_neurons, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Load and prepare the Fashion-MNIST dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1)).astype('float32') / 255
testX = testX.reshape((testX.shape[0], 28, 28, 1)).astype('float32') / 255
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# Define the parameter ranges to test
num_filters_options = [32, 64, 128]
kernel_size_options = [(3, 3), (5, 5)]
dense_neurons_options = [32, 64, 128]
learning_rate_options = [0.01, 0.001]
epochs = 10

# Function to run experiments
def run_experiments():
    results = []

    # Vary number of filters
    for num_filters in num_filters_options:
        model = create_model(num_filters=num_filters)
        model.fit(trainX, trainY, epochs=epochs, verbose=0)
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        results.append(('num_filters', num_filters, accuracy))
    
    # Vary kernel size
    for kernel_size in kernel_size_options:
        model = create_model(kernel_size=kernel_size)
        model.fit(trainX, trainY, epochs=epochs, verbose=0)
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        results.append(('kernel_size', kernel_size, accuracy))
    
    # Vary dense layer neurons
    for dense_neurons in dense_neurons_options:
        model = create_model(dense_neurons=dense_neurons)
        model.fit(trainX, trainY, epochs=epochs, verbose=0)
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        results.append(('dense_neurons', dense_neurons, accuracy))
    
    # Vary learning rate
    for learning_rate in learning_rate_options:
        model = create_model(learning_rate=learning_rate)
        model.fit(trainX, trainY, epochs=epochs, verbose=0)
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        results.append(('learning_rate', learning_rate, accuracy))

    # Print results
    for param, value, accuracy in results:
        print(f"{param} = {value}, Accuracy = {accuracy:.4f}")

run_experiments()
