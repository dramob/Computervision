from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.optimizers import SGD
import time

# Load and preprocess the dataset
(trainX, trainY), (testX, testY) = fashion_mnist.load_data()
trainX = trainX.reshape(-1, 28, 28, 1).astype('float32') / 255.0
testX = testX.reshape(-1, 28, 28, 1).astype('float32') / 255.0
trainY = to_categorical(trainY, 10)
testY = to_categorical(testY, 10)

# Define a function to create the model
def define_model(n_neurons):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(n_neurons, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Neurons in the dense layer to evaluate
neurons_list = [16, 64, 128, 256, 512]

# Store the results
results = []

for n_neurons in neurons_list:
    # Start the clock
    start_time = time.time()

    # Create and train the model
    model = define_model(n_neurons)
    model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=2)

    # Evaluate the model
    _, accuracy = model.evaluate(testX, testY, verbose=0)

    # Stop the clock
    end_time = time.time()

    # Append to results
    results.append((n_neurons, accuracy, end_time - start_time))

# Display a summary of results
print("\nSummary of Results:")
for n_neurons, accuracy, time_taken in results:
    print(f'Number of Neurons: {n_neurons}, Accuracy: {accuracy:.3f}, Training Time: {time_taken:.2f}s')
