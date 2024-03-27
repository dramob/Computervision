import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt

# Function to create the baseline model
def baseline_model(num_pixels, num_classes, hidden_neurons=8):
    model = Sequential()
    model.add(Dense(hidden_neurons, input_dim=num_pixels, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Function to load the model, make predictions, and display images
def load_model_and_predict(weights_file, num_pixels, num_classes):
    # Initialize model
    model = baseline_model(num_pixels, num_classes)

    # Load the saved weights
    model.load_weights(weights_file)

    # Load MNIST test dataset
    (_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Reshape and normalize the test data
    x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32') / 255

    # Make predictions on the first 5 images
    predictions = model.predict(x_test[:5])
    predicted_classes = np.argmax(predictions, axis=1)

    # Display the images and predictions
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[i].reshape(28, 28), cmap=plt.get_cmap('gray'))
        plt.title(f"Predicted: {predicted_classes[i]}")
        plt.axis('off')
        # Save the subplot as an image file
        plt.savefig(f'image_{i}.png')
    plt.show()

# Main function to execute the script
def main():
    num_pixels = 784  # 28*28 pixels
    num_classes = 10  # 10 digits
    weights_file = 'model_weights.weights.h5'  # File to save weights

    # Load the model and make predictions
    load_model_and_predict(weights_file, num_pixels, num_classes)

if __name__ == "__main__":
    main()
