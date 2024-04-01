#!/usr/bin/env python3

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16, Xception, InceptionV3, ResNet50, MobileNet
from tensorflow.keras import layers, models, optimizers

# Base directory for the dataset
base_dir = './Kaggle_Cats_And_Dogs_Dataset_Small'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Define image dimensions and batch size
IMG_WIDTH, IMG_HEIGHT = 150, 150
BATCH_SIZE = 20

# Function to load a pretrained model
def load_pretrained_model(model_name, input_shape):
    if model_name == 'VGG16':
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'Xception':
        base_model = Xception(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'InceptionV3':
        base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'ResNet50':
        base_model = ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif model_name == 'MobileNet':
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError("Unsupported model; choose from 'VGG16', 'Xception', 'InceptionV3', 'ResNet50', 'MobileNet'.")
    
    return base_model

# Function to add custom layers on top of a base model
def build_model(base_model):
    model = models.Sequential()
    model.add(base_model)
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    
    return model

# Function to compile and train the model
def compile_and_train_model(model, train_generator, validation_generator, epochs=10):
    model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(learning_rate=2e-5), metrics=['accuracy'])
    
    history = model.fit(train_generator,
                        steps_per_epoch=None,
                        epochs=epochs,
                        validation_data=validation_generator,
                        validation_steps=None,
                        )
    
    return model, history

# Setup data generators
def setup_data_generators(train_dir, validation_dir, img_width=IMG_WIDTH, img_height=IMG_HEIGHT, batch_size=BATCH_SIZE):
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(img_height, img_width), batch_size=batch_size, class_mode='binary')
    
    return train_generator, validation_generator

# Main execution function
def main():
    models_to_train = ['VGG16', 'Xception', 'InceptionV3', 'ResNet50', 'MobileNet']
    input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)
    epochs = 10  # Adjust based on your needs
    
    train_generator, validation_generator = setup_data_generators(train_dir, validation_dir)
    
    for model_name in models_to_train:
        print(f"Training model: {model_name}")
        base_model = load_pretrained_model(model_name, input_shape)
        base_model.trainable = False  # Freeze the base model
        model = build_model(base_model)
        model, history = compile_and_train_model(model, train_generator, validation_generator, epochs=epochs)
        # Optionally, save each model
        model.save(f'/content/{model_name}_cats_and_dogs.h5')
        # Clear session to free memory after each model training
        tf.keras.backend.clear_session()

if __name__ == '__main__':
    main()
