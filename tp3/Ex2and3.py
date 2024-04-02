import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from tensorflow.keras.models import load_model

# Function to load and preprocess an image
def load_and_preprocess_image(image_path, target_size=(150, 150)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return np.expand_dims(img_array, axis=0)  

# Load the model
model = load_model('./model_dropout=False_augmentation=False.h5')

# Exercise 2: Evaluate the model on a test set
test_dir = './Kaggle_Cats_And_Dogs_Dataset_Small/test'  # Replace with the path to your test directory
test_datagen = ImageDataGenerator(rescale=1./255)  # Assuming images need to be rescaled

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary'
)

test_loss, test_acc = model.evaluate(test_generator, steps=50)  # Adjust steps per your test set size
print(f'Test accuracy: {test_acc}')

# Exercise 3: Predict on new images
image_paths = ['test1.jpg', 'test2.jpg']  # Replace with paths to your images
images = np.vstack([load_and_preprocess_image(img_path) for img_path in image_paths])

predictions = model.predict(images)
print(predictions)
