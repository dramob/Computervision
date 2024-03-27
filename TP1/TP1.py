import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
#####################################################################################################################
#####################################################################################################################

# Load MNIST dataset
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()


#####################################################################################################################
#####################################################################################################################
def baseline_model(num_pixels, num_classes):
    # Initialize the sequential model
    model = Sequential()
      #EXERCISE1
    n=8
  

    # Define a hidden dense layer with n neurons
    model.add(Dense(n, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
    
    # Define the output dense layer
    model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    
    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictMLP(X_train, Y_train, X_test, Y_test):
    # Reshape the MNIST dataset from 2D to 1D vectors
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
    X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')
    
    # Normalize the input values
    X_train = X_train / 255
    X_test = X_test / 255
    
    # Transform the classes labels into a binary matrix
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    num_classes = Y_test.shape[1]
    
    # Build the model architecture
    model = baseline_model(num_pixels, num_classes)
    #Exercice 2
    # Train the model
    m=512
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=m, verbose=2)
    
    # Evaluate the model and compute the prediction error
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("Baseline Error: {:.2f}%".format(100 - scores[1] * 100))

#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def CNN_model(input_shape, num_classes):

    # TODO - Application 2 - Step 5a - Initialize the sequential model
    model = None   #Modify this


    #TODO - Application 2 - Step 5b - Create the first hidden layer as a convolutional layer


    #TODO - Application 2 - Step 5c - Define the pooling layer


    #TODO - Application 2 - Step 5d - Define the Dropout layer


    #TODO - Application 2 - Step 5e - Define the flatten layer


    #TODO - Application 2 - Step 5f - Define a dense layer of size 128


    #TODO - Application 2 - Step 5g - Define the output layer


    #TODO - Application 2 - Step 5h - Compile the model


    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):

    #TODO - Application 2 - Step 2 - reshape the data to be of size [samples][width][height][channels]


    #TODO - Application 2 - Step 3 - normalize the input values from 0-255 to 0-1


    #TODO - Application 2 - Step 4 - One hot encoding - Transform the classes labels into a binary matrix


    #TODO - Application 2 - Step 5 - Call the cnn_model function
    model = None   #Modify this


    #TODO - Application 2 - Step 6 - Train the model


    #TODO - Application 2 - Step 8 - Final evaluation of the model - compute and display the prediction error


    return
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()

    # Preprocess data if necessary (e.g., normalization, reshaping)

    # Call the MLP training and prediction function
    trainAndPredictMLP(X_train, Y_train, X_test, Y_test)

    # Call the CNN training and prediction function
    trainAndPredictCNN(X_train, Y_train, X_test, Y_test)
    return
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
if __name__ == '__main__':
    main()
#####################################################################################################################
#####################################################################################################################
