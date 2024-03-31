from keras import layers
from keras import models
from keras.optimizers import rmsprop_v2
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
from keras.applications.vgg16 import VGG16

import os
import shutil

#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def visualizeTheTrainingPerformances(history):

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)
    pyplot.title('Training and validation accuracy')
    pyplot.plot(epochs, acc, 'bo', label = 'Training accuracy')
    pyplot.plot(epochs, val_acc, 'b', label = 'Validation accuracy')
    pyplot.legend()

    pyplot.figure()
    pyplot.title('Training and validation loss')
    pyplot.plot(epochs, loss, 'bo', label = 'Training loss')
    pyplot.plot(epochs, val_loss, 'b', label = 'Validation loss')
    pyplot.legend

    pyplot.show()

    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def prepareDatabase(original_directory, base_directory):

    #If the folder already exist remove everything
    if os.path.exists(base_directory):
        shutil.rmtree(base_directory)

    #Recreate the basefolder
    os.mkdir(base_directory)

    #TODO - Application 1 - Step 1a - Create the training folder in the base directory


    #TODO - Application 1 - Step 1b - Create the validation folder in the base directory


    #TODO - Application 1 - Step 1c - Create the test folder in the base directory


    #TODO - Application 1 - Step 1d - Create the cat/dog training/validation/testing directories - See figure 4

    # create the train_cats_directory


    # create the train_dogs_directory


    # create the validation_cats_directory


    # create the validation_dogs_directory


    # create the test_cats_directory


    # create the test_dogs_directory



    #TODO - Application 1 - Step 1e - Copy the first 1000 cat images into the training directory (train_cats_directory)



    #TODO - Application 1 - Step 1f - Copy the next 500 cat images into the validation directory (validation_cats_directory)



    #TODO - Application 1 - Step 1g  - Copy the next 500 cat images in to the test directory (test_cats_directory)



    # TODO - Application 1 - Step 1h - Copy the first 1000 dogs images into the training directory (train_dogs_directory)



    # TODO - Application 1 - Step 1i - Copy the next 500 dogs images into the validation directory (validation_dogs_directory)



    # TODO - Application 1 - Step 1j  - Copy the next 500 dogs images in to the test directory (test_dogs_directory)



    #TODO - Application 1 - Step 1k - As a sanitary check verify how many pictures are in each directory



    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineCNNModelFromScratch():

    #Application 1 - Step 3a - Initialize the sequential model
    model = models.Sequential()

    #TODO - Application 1 - Step 3b - Create the first hidden layer as a convolutional layer


    #TODO - Application 1 - Step 3c - Define a maxpooling layer


    #TODO - Application 1 - Step 3d - Create the third hidden layer as a convolutional layer


    #TODO - Application 1 - Step 3e - Define a pooling layer


    #TODO - Application 1 - Step 3f - Create another convolutional layer


    #TODO - Application 1 - Step 3g - Define a pooling layer


    #TODO - Application 1 - Step 3h - Create another convolutional layer


    #TODO - Application 1 - Step 3i - Define a pooling layer


    #TODO - Application 1 - Step 3j - Define the flatten layer


    #TODO - Application 1 - Step 3k - Define a dense layer of size 512


    #TODO - Application 1 - Step 3l - Define the output layer


    #TODO - Application 1 - Step 3m - Visualize the network arhitecture (list of layers)


    #TODO - Application 1 - Step 3n - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineCNNModelVGGPretrained():

    #TODO - Application 2 - Step 1 - Load the pretrained VGG16 network in a variable called baseModel
    #The top layers will be omitted; The input_shape will be kept to (150, 150, 3)


    #TODO - Application 2 - Step 2 -  Visualize the network arhitecture (list of layers)


    #TODO - Application 2 - Step 3 -  Freeze the baseModel convolutional layers in order not to allow training


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

    train_directory = base_directory + '/train'
    validation_directory = base_directory + '/validation'

    #TODO - Application 1 - Step 2 - Create the image data generators for train and validation



    #TODO - Application 1 - Step 2 - Analyze the output of the train and validation generators



    # return train_generator, validation_generator    #Uncomment this
#####################################################################################################################
#####################################################################################################################




#####################################################################################################################
#####################################################################################################################
def main():

    original_directory = "./Kaggle_Cats_And_Dogs_Dataset"
    base_directory = "./Kaggle_Cats_And_Dogs_Dataset_Small"

    #TODO - Application 1 - Step 1 - Prepare the dataset


    #TODO - Application 1 - Step 2 - Call the imagePreprocessing method


    #TODO - Application 1 - Step 3 - Call the method that creates the CNN model



    #TODO - Application 1 - Step 4 - Train the model



    #TODO - Application 1 - Step 5 - Visualize the system performance using the diagnostic curves



    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
