import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import time

# Function to create the baseline model with MSE as a metric
def baseline_model(num_pixels, num_classes, hidden_neurons):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Using categorical_crossentropy as the loss function and mse as an additional metric
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy', 'mse'])
    return model

# Function to train and predict with MLP and save the model weights
def train_and_predict_mlp(x_train, y_train, x_test, y_test, num_pixels, num_classes, batch_size):
    # Reshape and normalize input data
    x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32') / 255
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') / 255
    # One-hot encode the labels
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    # Initialize model
    model = baseline_model(num_pixels, num_classes, hidden_neurons=8)  # 8 neurons in hidden layer
    # Train model
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=batch_size, verbose=0)

    # Save model weights with the correct filename format
    model.save_weights('model_weights.weights.h5')

    # Evaluate the model
    loss, accuracy, mse = model.evaluate(x_test, y_test, verbose=0)
    print(f" Accuracy: {accuracy:.2f}%, MSE: {mse:.4f}")

# Main function
def main():
    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    num_pixels = 784  # 28*28 pixels in each image
    num_classes = 10  # 10 digits

    # Train the model with default parameters and save the weights
    train_and_predict_mlp(x_train, y_train, x_test, y_test, num_pixels, num_classes, batch_size=200)

# Run the main function
if __name__ == "__main__":
    main()
