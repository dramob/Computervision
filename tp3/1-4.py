#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

# Define the model architecture
# Exercise 4: Added a dropout layer to reduce overfitting
def defineCNNModelWithDropout():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(0.5),  # Dropout layer added here
        layers.Dense(512, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0
    return img_tensor

# Visualize training performance
# Exercise 1: Visualize and save training and validation accuracy and loss
def visualizeTheTrainingPerformance(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.savefig('training_validation_performance.png')  # Save the figure
    plt.show()

def main():
    base_dir = './Kaggle_Cats_And_Dogs_Dataset_Small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    # Data augmentation for Exercise 4
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

    model = defineCNNModelWithDropout()
    history = model.fit(
        train_generator,
        steps_per_epoch=16,  
        epochs=100,  # Exercise 4: Training for 100 epochs
        validation_data=validation_generator,
        validation_steps=16
    )

    # Exercise 1: Visualize training performance
    visualizeTheTrainingPerformance(history)

    # Exercise 2: Evaluate the model accuracy on the testing dataset
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    test_loss, test_acc = model.evaluate(test_generator, steps=50)
    print(f'Test accuracy: {test_acc}')

    # Exercise 3: Predict on new images
    image_paths = ['test1.jpg', 'test2.jpg']
    images = np.vstack([load_and_preprocess_image(img_path) for img_path in image_paths])
    predictions = model.predict(images)
    print(predictions)

  
    # Save outputs to a CSV file
    with open('model_output.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Test Accuracy', test_acc])
        writer.writerow(['Image Path', 'Prediction'])
        for img_path, prediction in zip(image_paths, predictions):
            writer.writerow([img_path, prediction[0]])

if __name__ == '__main__':
    main()
