# Application 1 - Step 1 - Import the dependencies
import numpy as np
from sklearn.model_selection import KFold
import keras
from keras.optimizers import  SGD
from keras.datasets import fashion_mnist
from keras.utils import to_categorical
from keras import layers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from matplotlib import pyplot
import cv2
from keras.models import Sequential
######################################################################
#####################################################################

########################################################################
########################################################################
def summarizeLearningCurvesPerformances(histories, accuracyScores):
    pyplot.figure(figsize=(12, 6))
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()
    print(f'Accuracy: mean={np.mean(accuracyScores)*100:.3f}%, std={np.std(accuracyScores)*100:.3f}%, n={len(accuracyScores)}')
##############################################################################
#############################################################################


##############################################################################
##############################################################################
def prepareData(trainX, trainY, testX, testY):
    trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
    testX = testX.reshape((testX.shape[0], 28, 28, 1))
    trainX = trainX.astype('float32') / 255.0
    testX = testX.astype('float32') / 255.0
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return trainX, trainY, testX, testY
#############################################################################
#############################################################################
#Exercice 1 
def display_first_nine_images(trainX, trainY):
    # Define the names of the classes in the Fashion-MNIST dataset
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # Create a window to display the images
    cv2.namedWindow('Fashion MNIST', cv2.WINDOW_NORMAL)

    # Display the first 9 images from the training dataset
    for i in range(9):
        img = trainX[i]
        label = trainY[i]
        img_resized = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
        cv2.putText(img_resized, class_names[label], (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('Fashion MNIST', img_resized)
        cv2.waitKey(0)  # Wait for a key press to show the next image

    cv2.destroyAllWindows()


###########################################################################
###########################################################################
def defineModel(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu', kernel_initializer='he_uniform'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])
    return model
#############################################################################
#############################################################################


#############################################################################
#############################################################################
def defineTrainAndEvaluateClassic(trainX, trainY, testX, testY):
    model = defineModel(trainX.shape[1:], trainY.shape[1])
    history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_split=0.1, verbose=1)
    _, accuracy = model.evaluate(testX, testY, verbose=0)
    print('Test accuracy:', accuracy)
    return history, accuracy
#############################################################################
#############################################################################


#############################################################################
#############################################################################
def defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY):
    k_folds = 5
    accuracyScores = []  # Ensure this is correctly initialized
    histories = []
    kfold = KFold(k_folds, shuffle=True, random_state=1)

    for train_idx, val_idx in kfold.split(trainX):
        # Split data
        x_train_fold, y_train_fold = trainX[train_idx], trainY[train_idx]
        x_val_fold, y_val_fold = trainX[val_idx], trainY[val_idx]

        # Define and compile the model
        model = defineModel(trainX.shape[1:], trainY.shape[1])

        # Fit the model
        history = model.fit(x_train_fold, y_train_fold, validation_data=(x_val_fold, y_val_fold), epochs=10, batch_size=32, verbose=0)

        # Evaluate the model
        _, accuracy = model.evaluate(testX, testY, verbose=0)
        print(f'Test accuracy: {accuracy}')

        # Append history and accuracy
        histories.append(history)
        accuracyScores.append(accuracy)  # This is crucial

    return histories, accuracyScores
##############################################################################
##############################################################################



##############################################################################
def main():
    # Load the dataset
    (trainX, trainY), (testX, testY) = fashion_mnist.load_data()
    print("Train data shape:", trainX.shape)
    print("Test data shape:", testX.shape)

    # Prepare data
    trainX, trainY, testX, testY = prepareData(trainX, trainY, testX, testY)

    # Define, train and evaluate the model classically
    history, accuracy = defineTrainAndEvaluateClassic(trainX, trainY, testX, testY)
    
    # Define, train and evaluate the model using K-Folds strategy
    histories, accuracyScores = defineTrainAndEvaluateKFolds(trainX, trainY, testX, testY)
    
    # System performance presentation
    summarizeLearningCurvesPerformances(histories, accuracyScores)
    

###########################################################################
###########################################################################





###########################################################################
###########################################################################
if __name__ == '__main__':
    main()
###########################################################################
###########################################################################

