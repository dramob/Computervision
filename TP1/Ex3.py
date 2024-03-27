import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import time

def create_baseline_model(num_pixels, num_classes, hidden_neurons):
    model = Sequential([
        Dense(hidden_neurons, input_dim=num_pixels, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
    return model

def preprocess_data(x_train, y_train, x_test, y_test, num_pixels, num_classes):
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') / 255
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    return x_train, y_train, x_test, y_test

def train_and_evaluate_mlp(x_train, y_train, x_test, y_test, num_pixels, num_classes, batch_size, hidden_neurons):
    model = create_baseline_model(num_pixels, num_classes, hidden_neurons)
    start_time = time.time()
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=batch_size, verbose=0)
    training_time = time.time() - start_time
    loss, accuracy, mse = model.evaluate(x_test, y_test, verbose=0)
    return accuracy, mse, training_time

def experiment_with_batch_sizes(x_train, y_train, x_test, y_test, num_pixels, num_classes, hidden_neurons):
    batch_sizes = [32, 64, 128, 256, 512]
    for batch_size in batch_sizes:
        print(f"\nTraining with batch size: {batch_size}")
        accuracy, mse, training_time = train_and_evaluate_mlp(x_train, y_train, x_test, y_test, num_pixels, num_classes, batch_size, hidden_neurons)
        print(f"Accuracy: {accuracy:.2f}%, MSE: {mse:.4f}, Training Time: {training_time:.2f} seconds")

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_pixels = 784
    num_classes = 10
    hidden_neurons = 8
    x_train, y_train, x_test, y_test = preprocess_data(x_train, y_train, x_test, y_test, num_pixels, num_classes)
    experiment_with_batch_sizes(x_train, y_train, x_test, y_test, num_pixels, num_classes, hidden_neurons)

if __name__ == "__main__":
    main()
