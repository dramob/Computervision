import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.utils import to_categorical

def cnn_model(input_shape, num_classes, kernel_size):
    model = Sequential([
        Conv2D(8, kernel_size, activation='relu', input_shape=input_shape),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate_cnn(x_train, y_train, x_test, y_test, input_shape, num_classes, kernel_size):
    model = cnn_model(input_shape, num_classes, kernel_size)
    model.fit(x_train, y_train, epochs=5, batch_size=200, verbose=2, validation_data=(x_test, y_test))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
    return accuracy

def main():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape((-1, 28, 28, 1)).astype('float32') / 255
    x_test = x_test.reshape((-1, 28, 28, 1)).astype('float32') / 255
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    filter_sizes = [(1, 1), (3, 3), (5, 5), (7, 7), (9, 9)]
    for size in filter_sizes:
        accuracy = train_and_evaluate_cnn(x_train, y_train, x_test, y_test, (28, 28, 1), 10, size)
        print(f"Filter Size: {size}, Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()
