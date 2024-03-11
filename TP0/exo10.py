import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam

# Function to define the updated ANN model with an additional hidden layer
def modelDefinition():
    model = Sequential()
    model.add(Dense(8, input_dim=13, kernel_initializer='normal', activation='relu'))
    # Additional hidden layer with 16 neurons
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, kernel_initializer='normal'))
    # Compile model
    model.compile(loss='mean_squared_error', optimizer=Adam())
    return model

# Main execution function
def main():
    # Load and preprocess the dataset
    data = pd.read_csv('Houses.csv').values
    np.random.shuffle(data)
    X = data[:, :-1]
    y = data[:, -1]
    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Get the model
    model = modelDefinition()
    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2)
    # Evaluate the model
    mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"Mean Square Error on the testing dataset: {mse}")

# Make sure to uncomment the following line when running in your own environment
    main()