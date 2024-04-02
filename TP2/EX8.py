import numpy as np
import tensorflow 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical

def preprocess_data(images, labels):
    images = images.reshape(-1, 28, 28, 1)
    images = images.astype('float32') / 255.0
    labels = to_categorical(labels)
    return images, labels

def create_model(input_shape=(28, 28, 1), num_classes=10):
    model = Sequential([
        Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_evaluate(model, train_images, train_labels, test_images, test_labels):
    model.fit(train_images, train_labels, epochs=10, batch_size=32, verbose=2, validation_data=(test_images, test_labels))
    loss, acc = model.evaluate(test_images, test_labels, verbose=0)
    print(f'Accuracy: {acc * 100:.2f}%')

def main():
    (train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

    train_images, train_labels = preprocess_data(train_images, train_labels)
    test_images, test_labels = preprocess_data(test_images, test_labels)

    model = create_model()
    train_and_evaluate(model, train_images, train_labels, test_images, test_labels)

if __name__ == "__main__":
    main()
