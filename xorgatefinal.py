import numpy as np
#####################################################################################################################
#####################################################################################################################


#####################################################################################################################
def tanh(n):
    return np.tanh(n)
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def tanhDerivative(n):
    return 1.0 - n**2
#####################################################################################################################
#####################################################################################################################



#####################################################################################################################
#####################################################################################################################
def forwardPropagationLayer(p, weights, biases):

    a = None  # the layer output

    # Multiply weights with the input vector (p) and add the bias   =>  n
    n = np.dot(p, weights) + biases

    # Pass the result to the activation function  =>  a
    a = tanh(n)

    return a
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
def mean_squared_error(y_true, y_pred):
    return (1/(2*len(y_true))) * np.sum((y_true - y_pred)**2)
#####################################################################################################################
#####################################################################################################################

def main():

    #Application 2 - Train a ANN in order to predict the output of an XOR gate.
    #The network should receive as input two values (0 or 1) and should predict the target output

    #Input data
    points = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1]
        ])

    #Labels
    labels = np.array([[0], [1], [1], [0]])

    # Initialize the weights and biases with random values
    inputSize = 2
    noNeuronsLayer1 = 2
    noNeuronsLayer2 = 1

    weightsLayer1 = np.random.uniform(size=(inputSize, noNeuronsLayer1))
    weightsLayer2 = np.random.uniform(size=(noNeuronsLayer1, noNeuronsLayer2))

    biasLayer1 = np.random.uniform(size=(1, noNeuronsLayer1))
    biasLayer2 = np.random.uniform(size=(1, noNeuronsLayer2))

    max_epochs = 5000
    learningRate = 0.3
    min_error = 0.01

    error = float('inf')
    epoch = 0

    # Train the network until the error is less than min_error or epochs reach max_epochs
    while error > min_error and epoch < max_epochs:
        epoch += 1

        # Forward Propagation
        hidden_layer_output = forwardPropagationLayer(points, weightsLayer1, biasLayer1)
        predicted_output = forwardPropagationLayer(hidden_layer_output, weightsLayer2, biasLayer2)

        # Backpropagation
        bkProp_error = labels - predicted_output
        d_predicted_output = bkProp_error * tanhDerivative(predicted_output)

        error_hidden_layer = d_predicted_output.dot(weightsLayer2.T)
        d_hidden_layer = error_hidden_layer * tanhDerivative(hidden_layer_output)

        # Updating Weights and Biases
        weightsLayer2 += hidden_layer_output.T.dot(d_predicted_output) * learningRate
        biasLayer2 += np.sum(d_predicted_output, axis=0, keepdims=True) * learningRate

        weightsLayer1 += points.T.dot(d_hidden_layer) * learningRate
        biasLayer1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * learningRate

        # Calculate error
        error = mean_squared_error(labels, predicted_output)

    print(f"Minimum number of epochs: {epoch}")
    print(f"Final error: {error}")

    # Print weights and bias
    print("weightsLayer1 = {}".format(weightsLayer1))
    print("biasesLayer1 = {}".format(biasLayer1))

    print("weightsLayer2 = {}".format(weightsLayer2))
    print("biasLayer2 = {}".format(biasLayer2))

    # Display the results
    for i in range(len(labels)):
        outL1 = forwardPropagationLayer(points[i], weightsLayer1, biasLayer1)
        outL2 = forwardPropagationLayer(outL1, weightsLayer2, biasLayer2)

        print("Input = {} - Predict = {} - Label = {}".format(points[i], outL2, labels[i]))
#####################################################################################################################
#####################################################################################################################

if __name__ == "__main__":
    main()
