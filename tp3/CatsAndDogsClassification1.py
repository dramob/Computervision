import tensorflow
from tensorflow.keras import layers
from tensorflow.keras import models
#from tensorflow.keras.optimizers import rmsprop_v2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from tensorflow.keras.applications import VGG16,ResNet50,InceptionV3,Xception,MobileNet
from tensorflow.keras.optimizers import RMSprop 
import os
import shutil
from glob import glob

#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def visualizeTheTrainingPerformances(history,model_name):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    pyplot.title('Training and validation accuracy')
    pyplot.plot(epochs, acc, 'bo', label = 'Training accuracy')
    pyplot.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
    pyplot.legend()
    pyplot.savefig('Accuracy '+model_name+'.png')
    pyplot.figure()
    pyplot.title('Training and validation loss')
    pyplot.plot(epochs, loss, 'bo', label = 'Training loss')
    pyplot.plot(epochs, val_loss, 'b', label = 'Validation loss')
    pyplot.legend
    pyplot.savefig('Loss '+model_name+'.png')
    pyplot.show()

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def prepareDatabase(original_directory, base_directory):
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)
    os.mkdir(base_directory)

    train_directory = os.path.join(base_directory, 'train')
    os.mkdir(train_directory)
    validation_directory = os.path.join(base_directory, 'validation')
    os.mkdir(validation_directory)
    test_directory = os.path.join(base_directory, 'test')
    os.mkdir(test_directory)

    directories = [(train_directory, 1000), (validation_directory, 500), (test_directory, 500)]
    for animal in ['cats', 'dogs']:
        for directory, count in directories:
            animal_directory = os.path.join(directory, animal)
            os.mkdir(animal_directory)
            original_animal_directory = os.path.join(original_directory, animal)
            fnames = [f'{i}.jpg' for i in range(count)]
            for fname in fnames:
                src = os.path.join(original_animal_directory, fname)
                dst = os.path.join(animal_directory, fname)
                shutil.copyfile(src, dst)

    print('Dataset preparation is completed.')



#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineCNNModelFromScratch():
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
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # Update here: change 'lr' to 'learning_rate'
    model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=1e-4), metrics=['accuracy'])
    model.summary()

    return model

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineCNNModelVGGPretrained():

    #TODO - Application 2 - Step 1 - Load the pretrained VGG16 network in a variable called baseModel
    #The top layers will be omitted; The input_shape will be kept to (150, 150, 3)
	#baseModel = VGG16(input_shape=(150, 150, 3), include_top=False, weights="imagenet")

    #TODO - Application 2 - Step 2 -  Visualize the network arhitecture (list of layers)


    #TODO - Application 2 - Step 3 -  Freeze the baseModel convolutional layers in order not to allow training
	#for layer in baseModel.layers:
        #layer.trainable = False

    #baseModel.summary()

    #TODO - Application 2 - Step 4 - Create the final model and add the layers from the baseModel
    VGG_model = models.Sequential()
    #VGG_model.add(baseModel)            #Uncomment this


    # TODO - Application 2 - Step 4a - Add the flatten layer


    # TODO - Application 2 - Step 4b - Add the dropout layer


    # TODO Application 2 - Step 4c - Add a dense layer of size 512


    # TODO - Application 2 - Step 4d - Add the output layer


    # TODO - Application 2 - Step 4e - Compile the model


    return VGG_model
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
def imagePreprocessing(base_directory):
    train_dir = os.path.join(base_directory, 'train')
    validation_dir = os.path.join(base_directory, 'validation')
    test_dir = os.path.join(base_directory, 'test')

    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=40, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True,)
    test_datagen = ImageDataGenerator(rescale=1./255)  # Only rescaling for validation and test sets

    train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150, 150), batch_size=20, class_mode='binary')
    validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150, 150), batch_size=20, class_mode='binary')

    return train_generator, validation_generator

#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
def main():

    original_directory = "./Kaggle_Cats_And_Dogs_Dataset"
    base_directory = "./Kaggle_Cats_And_Dogs_Dataset_Small"


   # prepareDatabase(original_directory, base_directory)
    
    train_generator, validation_generator = imagePreprocessing(base_directory)
    model = defineCNNModelFromScratch()

    history = model.fit(train_generator, steps_per_epoch=100, epochs=30, validation_data=validation_generator, validation_steps=50)

    visualizeTheTrainingPerformances(history, "CNN_From_Scratch")

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
