import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# Define the extended CNN Model
def create_extended_cnn_model(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(30, kernel_size=(5, 5), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to train the model
def train_model(model, x_train, y_train, x_test, y_test, epochs, batch_size):
    model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(x_test, y_test))
    return model

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

    # Train and evaluate the extended CNN
    model = create_extended_cnn_model(input_shape, num_classes)
    trained_model = train_model(model, x_train, y_train, x_test, y_test, epochs=5, batch_size=200)
    accuracy = evaluate_model(trained_model, x_test, y_test)
    print(f"Extended CNN Architecture Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
