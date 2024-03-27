import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import time

# Load Fashion-MNIST dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()

# Normalize input values
trainX = trainX.astype('float32') / 255.0
testX = testX.astype('float32') / 255.0

# Reshape data
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)

# One-hot encode the target variables
trainY = to_categorical(trainY, num_classes=10)
testY = to_categorical(testY, num_classes=10)

def define_train_and_evaluate_classic(trainX, trainY, testX, testY, num_filters):
    model = Sequential([
        Conv2D(num_filters, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(16, activation='relu', kernel_initializer='he_uniform'),
        Dense(10, activation='softmax')
    ])

    # Compile the model
    optimizer = SGD(learning_rate=0.01, momentum=0.9)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    start_time = time.time()
    model.fit(trainX, trainY, epochs=5, batch_size=32, verbose=0)
    end_time = time.time()
    training_time = end_time - start_time

    # Evaluate the model
    _, test_accuracy = model.evaluate(testX, testY, verbose=0)

    return test_accuracy, training_time

num_filters_list = [8, 16, 32, 64, 128]
results = {}

# Evaluate each configuration
for num_filters in num_filters_list:
    accuracy, training_time = define_train_and_evaluate_classic(trainX, trainY, testX, testY, num_filters)
    results[num_filters] = (accuracy, training_time)

# Print results
print("Results:")
print("Number of Filters\tAccuracy\tTraining Time (s)")
for num_filters, (accuracy, training_time) in results.items():
    print(f"{num_filters}\t\t\t{accuracy:.4f}\t\t{training_time:.2f}")
