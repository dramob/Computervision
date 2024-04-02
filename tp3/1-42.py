#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import csv
from matplotlib import pyplot as plt

# Define model architectures
def define_model(with_dropout=False):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten()
    ])
    
    if with_dropout:
        model.add(layers.Dropout(0.5))  # Add dropout layer if with_dropout is True

    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Data Preprocessing: Setup data generators with and without data augmentation
def setup_data_generators(use_augmentation=False):
    if use_augmentation:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
    else:
        train_datagen = ImageDataGenerator(rescale=1./255)  # Without data augmentation
    
    validation_datagen = ImageDataGenerator(rescale=1./255)  # Always rescale

    # Assuming the directory structure provided earlier
    train_generator = train_datagen.flow_from_directory(
        './Kaggle_Cats_And_Dogs_Dataset_Small/train',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )
    validation_generator = validation_datagen.flow_from_directory(
        './Kaggle_Cats_And_Dogs_Dataset_Small/validation',
        target_size=(150, 150),
        batch_size=20,
        class_mode='binary'
    )

    return train_generator, validation_generator

# Visualize training performance
def visualizeTheTrainingPerformance(history, title_suffix):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy ' + title_suffix)
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss ' + title_suffix)
    plt.legend()
    
    plt.savefig('performance_' + title_suffix.replace(" ", "_") + '.png')
    plt.show()

# Main function to run the experiments
def run_experiments():
    configurations = [(True, True), (True, False), (False, True), (False, False)]
    
    for with_dropout, use_augmentation in configurations:
        print(f"Running experiment: Dropout={with_dropout}, Data Augmentation={use_augmentation}")
        model = define_model(with_dropout=with_dropout)
        train_generator, validation_generator = setup_data_generators(use_augmentation=use_augmentation)
        
        # Calculate steps based on the actual dataset size

        # Run training
        history = model.fit(
            train_generator,
            epochs=100,  
            steps_per_epoch=60,
            validation_data=validation_generator,
            validation_steps=50
        )
        
        title_suffix = f"Dropout={with_dropout} Augmentation={use_augmentation}"
        visualizeTheTrainingPerformance(history, title_suffix)
        model.save(f'model_dropout={with_dropout}_augmentation={use_augmentation}.h5')
        del model 
        tf.keras.backend.clear_session()
if __name__ == '__main__':
    run_experiments()
