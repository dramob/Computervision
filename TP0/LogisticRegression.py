import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Define the ANN model
def modelDefinition():
    model = Sequential()
    model.add(Dense(8, input_dim=13, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def main():
    # Load the dataset
    data = pd.read_csv("houses.csv")

    # Shuffle the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split the data
    X = data.drop('medValue', axis=1).values
    y = data['medValue'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define the model
    model = modelDefinition()

    # Train the model
    model.fit(X_train, y_train, epochs=100, batch_size=16, verbose=2)

    # Make predictions
    predictions = model.predict(X_test)

    # Compute the Mean Squared Error
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Square Error: {mse}")

if __name__ == '__main__':
    main()












    






