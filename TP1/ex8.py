import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import time

# Define the CNN Model with 128 neurons in the dense layer
def create_cnn_model(input_shape, num_classes, kernel_size=(3, 3)):
    model = Sequential()
    model.add(Conv2D(8, kernel_size=kernel_size, activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  # 128 neurons in the dense layer
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, x_train, y_train, x_test, y_test, epochs):
    start_time = time.time()
    model.fit(x_train, y_train, epochs=epochs, batch_size=200, verbose=2, validation_data=(x_test, y_test))
    training_time = time.time() - start_time
    return training_time

# Function to evaluate the model
def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

# Function to load and preprocess MNIST dataset
def load_and_preprocess_mnist():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    return x_train, y_train, x_test, y_test

# Main function
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_mnist()
    input_shape = (28, 28, 1)
    num_classes = 10

    # Experiment with different numbers of epochs
    epoch_values = [1, 2, 5, 10, 20]
    for epochs in epoch_values:
        model = create_cnn_model(input_shape, num_classes)
        training_time = train_model(model, x_train, y_train, x_test, y_test, epochs)
        accuracy = evaluate_model(model, x_test, y_test)
        print(f"Epochs: {epochs}, Accuracy: {accuracy * 100:.2f}%, Training Time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main()
