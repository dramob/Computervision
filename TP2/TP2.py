# Application 1 - Step 1 - Import the dependencies
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import gradient_descent_v2
from keras.datasets import fashion_mnist
from keras.utils import np_utils
from keras import layers
from matplotlib import pyplot
import cv2
#####################################################################################################################
#####################################################################################################################

#####################################################################################################################
#####################################################################################################################
def summarizeLearningCurvesPerformances(histories, accuracyScores):

    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(211)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='green', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='red', label='test')

        # plot accuracy
        pyplot.subplot(212)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='green', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='red', label='test')

        #print accuracy for each split
        print("Accuracy for set {} = {}".format(i, accuracyScores[i]))

    pyplot.show()

    print('Accuracy: mean = {:.3f} std = {:.3f}, n = {}'.format(np.mean(accuracyScores) * 100, np.std(accuracyScores) * 100, len(accuracyScores)))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def prepareData(trainX, trainY, testX, testY):

    #TODO - Application 1 - Step 4a - reshape the data to be of size [samples][width][height][channels]


    #TODO - Application 1 - Step 4b - normalize the input values


    #TODO - Application 1 - Step 4c - Transform the classes labels into a binary matrix


    return trainX, trainY, testX, testY
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):

    #TODO - Application 1 - Step 6a - Initialize the sequential model
    model = None   #Modify this

    #TODO - Application 1 - Step 6b - Create the first hidden layer as a convolutional layer


    #TODO - Application 1 - Step 6c - Define the pooling layer


    #TODO - Application 1 - Step 6d - Define the flatten layer


    #TODO - Application 1 - Step 6e - Define a dense layer of size 16


    #TODO - Application 1 - Step 6f - Define the output layer


    #TODO - Application 1 - Step 6g - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):

    #TODO - Application 1 - Step 6 - Call the defineModel function


    #TODO - Application 1 - Step 7 - Train the model


    #TODO - Application 1 - Step 8 - Evaluate the model


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):

    k_folds = 5

    accuracyScores = []
    histories = []

    #Application 2 - Step 2 - Prepare the cross validation datasets
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(trainX):

        #TODO - Application 2 - Step 3 - Select data for train and validation


        #TODO - Application 2 - Step 4 - Build the model - Call the defineModel function


        #TODO - Application 2 - Step 5 - Fit the model


        #TODO - Application 2 - Step 6 - Save the training related information in the histories list


        #TODO - Application 2 - Step 7 - Evaluate the model on the test dataset


        #TODO - Application 2 - Step 8 - Save the accuracy in the accuracyScores list

        pass #DELETE THIS!

    return histories, accuracyScores
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    #TODO - Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras


    #TODO - Application 1 - Step 3 - Print the size of the train/test dataset


    #TODO - Application 1 - Step 4 - Call the prepareData method


    #TODO - Application 1 - Step 5 - Define, train and evaluate the model in the classical way


    #TODO - Application 2 - Step 1 - Define, train and evaluate the model using K-Folds strategy


    #TODO - Application 2 - Step9 - System performance presentation


    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
