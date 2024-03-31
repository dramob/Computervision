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

    #Application 1 - Step 3a - Initialize the sequential model
    model = models.Sequential()

    #TODO - Application 1 - Step 3b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(150, 150, 3),activation='relu'))

    #TODO - Application 1 - Step 3c - Define a maxpooling layer
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))

    #TODO - Application 1 - Step 3d - Create the third hidden layer as a convolutional layer


    #TODO - Application 1 - Step 3e - Define a pooling layer


    #TODO - Application 1 - Step 3f - Create another convolutional layer


    #TODO - Application 1 - Step 3g - Define a pooling layer


    #TODO - Application 1 - Step 3h - Create another convolutional layer


    #TODO - Application 1 - Step 3i - Define a pooling layer


    #TODO - Application 1 - Step 3j - Define the flatten layer
    model.add(layers.Flatten())
    #model.add(layers.Dropout(rate=0.5))
    #TODO - Application 1 - Step 3k - Define a dense layer of size 512
    model.add(layers.Dense(512, activation='relu'))

    #TODO - Application 1 - Step 3l - Define the output layer
    model.add(layers.Dense(1, activation='sigmoid'))

    #TODO - Application 1 - Step 3m - Visualize the network arhitecture (list of layers)
    model.summary()


    #TODO - Application 1 - Step 3n - Compile the model
    model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

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

    prepareDatabase(original_directory, base_directory)


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
