 #!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Define the model architecture
def defineCNNModelWithDropout():
    # Exercise 4: Adding a dropout layer to reduce overfitting
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))  # Dropout layer added here
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def load_and_preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255.0  # Model expects input in this range
    return img_tensor


def main():

    base_dir = './Kaggle_Cats_And_Dogs_Dataset_Small'
    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')


    train_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

    # Define the model with dropout to reduce overfitting (Exercise 4)
    model = defineCNNModelWithDropout()

    # Train the model for 100 epochs (Exercise 1)
    history = model.fit(train_generator, steps_per_epoch=100, epochs=100, validation_data=validation_generator, validation_steps=50)

    # Save the model (Exercise 3)
    model.save('Model_cats_dogs_small_dataset.h5')

    # Evaluate the model accuracy on the testing dataset (Exercise 2)
    test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    test_loss, test_acc = model.evaluate(test_generator, steps=50)
    print(f'Test accuracy: {test_acc}')

    # Make a prediction on new images (Exercise 3)
    image_paths = ['test1.jpg', 'test2.jpg']
    images = np.vstack([load_and_preprocess_image(img_path) for img_path in image_paths])
    predictions = model.predict(images)
    print(predictions)

if __name__ == '__main__':
    main()
