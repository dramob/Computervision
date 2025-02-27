# Application 1 - Step 1 - Import the dependencies
import numpy as np
import keras
from sklearn.model_selection import KFold
import tensorflow as tf
from keras.optimizers import SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
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

    # Application 1 - Step 4a - reshape the data to be of size [samples][width][height][channels]
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))

    # Application 1 - Step 4b - normalize the input values
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0

    # Application 1 - Step 4c - Transform the classes labels into a binary matrix
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)

    return trainX, trainY, testX, testY
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineModel(input_shape, num_classes):

    # Application 1 - Step 6a - Initialize the sequential model
    model = keras.Sequential()

    # Application 1 - Step 6b - Create the first hidden layer as a convolutional layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape))

    # Application 1 - Step 6c - Define the pooling layer
    model.add(layers.MaxPooling2D((2, 2)))

    # Application 1 - Step 6d - Define the flatten layer
    model.add(layers.Flatten())

    # Application 1 - Step 6e - Define a dense layer of size 16
    model.add(layers.Dense(16, activation='relu', kernel_initializer='he_uniform'))

    # Application 1 - Step 6f - Define the output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    # Application 1 - Step 6g - Compile the model
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):

    # Application 1 - Step 6 - Call the defineModel function
    model = defineModel((28, 28, 1), 10)

    # Application 1 - Step 7 - Train the model
    history = model.fit(trainX, trainY, epochs=5, batch_size=32, validation_data=(testX, testY), verbose=2)

    # Application 1 - Step 8 - Evaluate the model
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print(f'Accuracy: {accuracy*100:.2f}%')

    return history, accuracy
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
        # Application 2 - Step 3 - Select data for train and validation
        trainX_fold, trainY_fold = trainX[train_idx], trainY[train_idx]
        valX_fold, valY_fold = trainX[val_idx], trainY[val_idx]

        # Application 2 - Step 4 - Build the model - Call the defineModel function
        model = defineModel((28, 28, 1), 10)

        # Application 2 - Step 5 - Fit the model
        history = model.fit(trainX_fold, trainY_fold, epochs=5, batch_size=32, validation_data=(valX_fold, valY_fold), verbose=2)

        # Application 2 - Step 6 - Save the training related information in the histories list
        histories.append(history)

        # Application 2 - Step 7 - Evaluate the model on the test dataset
        _, accuracy = model.evaluate(testX, testY, verbose=0)

        # Application 2 - Step 8 - Save the accuracy in the accuracyScores list
        accuracyScores.append(accuracy)

    return histories, accuracyScores
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def main():

    # Application 1 - Step 2 - Load the Fashion MNIST dataset in Keras
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()

    # Application 1 - Step 3 - Print the size of the train/test dataset
    print(f'Training data shape: {trainX.shape}, Training labels shape: {trainY.shape}')
    print(f'Test data shape: {testX.shape}, Test labels shape: {testY.shape}')

    # Application 1 - Step 4 - Call the prepareData method
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)

    # Application 1 - Step 5 - Define, train and evaluate the model in the classical way
    history, accuracy = defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)

    # Application 2 - Step 1 - Define, train and evaluate the model using K-Folds strategy
    histories, accuracyScores = defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY)

    # Application 2
