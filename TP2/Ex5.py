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

def define_model(learning_rate):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(16, activation='relu'),
        Dense(10, activation='softmax')
    ])
    optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

learning_rates = [0.1, 0.01, 0.001, 0.0001, 0.00001]
results = []

for lr in learning_rates:
    start_time = time.time()
    model = define_model(lr)
    model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=2)
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    end_time = time.time()
    print(f'Learning Rate: {lr}, Test Accuracy: {accuracy:.3f}, Time: {end_time - start_time:.2f}s')
    results.append((lr, accuracy, end_time - start_time))

print("\nSummary of Results:")
for lr, accuracy, time_taken in results:
    print(f'Learning Rate: {lr}, Accuracy: {accuracy:.3f}, Training Time: {time_taken:.2f}s')
