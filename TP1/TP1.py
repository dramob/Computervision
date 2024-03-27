import keras
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense ,Conv2D, MaxPooling2D, Dropout, Flatten
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
    model = Sequential()
    #EXERCICE6
    n=1
    model.add(Conv2D(32, kernel_size=(n, n), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def trainAndPredictCNN(X_train, Y_train, X_test, Y_test):
    # Reshape data to fit the CNN input requirements
    X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
    
    # Normalize the input values
    X_train = X_train / 255
    X_test = X_test / 255
    
    # One-hot encoding
    Y_train = to_categorical(Y_train)
    Y_test = to_categorical(Y_test)
    num_classes = Y_test.shape[1]
    
    # Initialize the CNN model
    model = CNN_model(X_train.shape[1:], num_classes)
    
    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=200, verbose=2)
    
    # Evaluate the model
    scores = model.evaluate(X_test, Y_test, verbose=0)
    print("CNN Error: {:.2f}%".format(100 - scores[1] * 100))
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
#####################################################################################################################
def main():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()


    # Call the MLP training and prediction function (uncomment to run)
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
