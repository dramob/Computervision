#####################################################################################################################
#####################################################################################################################
import numpy as np
def activationFunction(n):
    #return 1 if n >= 0 else 0
    return 1 / (1 + np.exp(-n))
#####################################################################################################################
#####################################################################################################################

def forwardPropagation(p, weights, bias):
    # Application 1 - Step 4a - Multiply weights with the input vector (p) and add the bias
    n = sum(p[i] * weights[i] for i in range(len(p))) + bias
    # Application 1 - Step 4c - Pass the result to the activation function
    a = activationFunction(n)
    return a

#####################################################################################################################
#####################################################################################################################

def main():
    # Application 1 - Train a single neuron perceptron in order to predict the output of an AND gate.

    # Input data
    P = [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ]

    # Labels
    t = [0, 1, 1, 1]

    # Application 1 - Step 2 - Initialize the weights and bias with zero
    weights = [0, 0]
    bias = 0

    # Application 1 - Step 3 - Set the number of training steps (epochs)
    epochs = 5

    # Application 1 - Step 4 - Perform the neuron training for multiple epochs
    for ep in range(epochs):
        for i in range(len(P)):
            # Call the forwardPropagation method
            a = forwardPropagation(P[i], weights, bias)
            # Compute the prediction error
            error = t[i] - a
            # Update the weights
            weights[0] = weights[0] + error * P[i][0]
            weights[1] = weights[1] + error * P[i][1]
            # Update the bias
            bias = bias + error

    # Application 1 - Step 8 - Print weights and bias
    print("Final weights:", weights)
    print("Final bias:", bias)

    # Application 1 - Step 9 - Display the results
    for i in range(len(P)):
        predicted = forwardPropagation(P[i], weights, bias)
        print(f"Input: {P[i]}, Predicted: {predicted}, Actual: {t[i]}")

#####################################################################################################################
#####################################################################################################################

if __name__ == "__main__":
    main()
####################################################################################################################

