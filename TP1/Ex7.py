import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical
import time

def create_cnn_model(input_shape, num_classes, num_neurons, kernel_size=(3, 3)):
    model = Sequential([
        Conv2D(8, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(num_neurons, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_cnn(x_train, y_train, x_test, y_test, input_shape, num_classes, num_neurons):
    start_time = time.time()
    model = create_cnn_model(input_shape, num_classes, num_neurons)
    model.fit(x_train, y_train, epochs=5, batch_size=200, verbose=2, validation_data=(x_test, y_test))
    training_time = time.time() - start_time
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy, training_time

def main():
    # Load and preprocess MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    # Experiment with different numbers of neurons in the dense layer
    neuron_counts = [16, 64, 128, 256, 512]
    for neurons in neuron_counts:
        accuracy, training_time = train_and_evaluate_cnn(x_train, y_train, x_test, y_test, (28, 28, 1), 10, neurons)
        print(f"Neurons: {neurons}, Accuracy: {accuracy * 100:.2f}%, Training Time: {training_time:.2f} seconds")

if __name__ == "__main__":
    main()
