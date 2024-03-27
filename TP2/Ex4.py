import time
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load and preprocess the Fashion MNIST dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((-1, 28, 28, 1)).astype('float32') / 255.0
trainY = to_categorical(trainY)
testY = to_categorical(testY)

def define_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

epochs_list = [1, 2, 5, 10, 20]
results = []

for n_epochs in epochs_list:
    start_time = time.time()
    model = define_model()
    model.fit(trainX, trainY, epochs=n_epochs, batch_size=32, validation_data=(testX, testY), verbose=2)
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    end_time = time.time()
    print(f'Epochs: {n_epochs}, Test Accuracy: {accuracy:.3f}, Time: {end_time - start_time:.2f}s')
    results.append((n_epochs, accuracy, end_time - start_time))

print("\nSummary of Results:")
for n_epochs, accuracy, time_taken in results:
    print(f'Number of Epochs: {n_epochs}, Accuracy: {accuracy:.3f}, Training Time: {time_taken:.2f}s')
