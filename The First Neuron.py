import math

import numpy as np

###################################
#    Coding our first neurons     #
###################################

# Let's assume a layer with 4 inputs and 3 neurons

# For a single neuron, lets assume that we have 4 inputs
inputs = [1,2,3, 2.5]

# Lets also initialize some random weights and a random bias for each neuron
weights = [[0.2,0.8,-0.5, 1.0], [0.5,-0.91,0.26, -0.5], [-0.26,-0.27,0.17, 0.87]]
biases = [2,3,0.5]

# initialize output list
layer_output = []

# For each neuron (zip allows us to iterate through both lists)
for neuron_weights,neuron_bias in zip(weights,biases):

    # Non activated
    neuron_output = 0

    # Sum neuron input * weights
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += weight * n_input

    # Add bias
    neuron_output += neuron_bias

    # Add neuron results into an output list
    layer_output.append(neuron_output)

print("For one layer, these are the neuron outputs: ", layer_output)

############################################
#    Simplifying our code: Dot product     #
############################################

# Our sum of weights operation can be conducted using the dot product of vectors
inputs = [1.0,2.0,3.0,2.5]
weights = [[0.2,0.8,-0.5,1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17,0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights,inputs) + biases

print("Using the Numpy dot product: ", outputs)

#####################################################
#    Taking it a step further: Matrix operations    #
#####################################################

# input consists of 4 nodes, with 3 samples (Batching)
inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

biases = [2.0,3.0,0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases

print(outputs)

#####################################################
#                   Adding Layers                   #
#####################################################

# input consists of 4 nodes, with 3 samples (Batching)
inputs = [[1.0,2.0,3.0,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

# Layer 1 weights
weights = [[0.2,0.8,-0.5,1.0],
           [0.5,-0.91,0.26,-0.5],
           [-0.26,-0.27,0.17,0.87]]

# Layer 1 biases
biases = [2.0,3.0,0.5]

# Layer 2 weights (3 neurons
weights2 = [[0.1, -0.14,0.5],
            [-0.5,0.12,-0.33],
            [-0.44,0.73,-0.13]]

# Layer 2 biases
biases2 = [-1,2,-0.5]

# First, lets calculate the outputs for layer 1
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

# Now we can calculate the final output
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

#####################################################
#   Basic example of the ReLu activation function   #
#####################################################

input_example = [0,2,-1,3.3,-2.7,1.1,2.2,-100]
output_example = []

for i in input_example:
    if i<=0:
        output_example.append(0)
    else:
        output_example.append(i)

print(output_example)

#####################################################
#          The Softmax activation function          #
#####################################################

# Lets first assume we now have layer output data
layer_outputs_example = [4.8,1.21,2.385]

# First we exponentiate the outputs (math.e). THis is to remove non-negative values to calcualte probabilities
E = math.e

exp_values = []

for output in layer_outputs_example:
    exp_values.append(E**output)

# We now want to convert to a probability distribution

# Normalize
norm_base = sum(exp_values)
norm_values = []
for val in exp_values:
    norm_values.append(val/norm_base)

#####################################################
#          Categorical Cross-Entropy Loss           #
#####################################################

#####################################################
#                Using Training Data                #
#####################################################
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

nnfs.init()

X,y = spiral_data(samples = 100, classes = 3)

# Example of training data
"""plt.scatter(X[:,0],X[:,1])
plt.show()"""

# Example of classified output, where our data has been classified into three spirals (0, 1 and 2)
"""plt.scatter(X[:,0],X[:,1], c=y, cmap='brg')
plt.show()"""

#####################################################
#         Setting up the Dense Layer Class          #
#####################################################

nnfs.init()

# Lets use the training data introduced above
X,y = spiral_data(samples = 100, classes = 3)

# Dense layer
class Layer_Dense:

    # Initialize the layer. This function is called every time an object is created usign the class
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases. Let's make the random to start
        # Note that we have already "transposed" here: (inputs, neurons) vs (neurons, inputs)
        # Randn produces a gaussian distribution with mean - and variance 1. It will generate random numbers centered around 0
        # We first multiply by 0.01 to make these generated numbers much smaller
        # The idea; start a model with non-zero values small enough that they won't influence training
        self.weights = 0.01 * np.random.randn(n_inputs,n_neurons)
        # Starting with an inital bias of 0. Here, the biases are a row vector
        self.biases = np.zeros((1,n_neurons))


    def forward(self,inputs):
        # Calculate the output values from inputs, weights and biases
        # We want to update the output here.
        self.output = np.dot(inputs, self.weights) + biases

# Our ReLu activation function
class Activation_ReLu:

    def forward(self, inputs):

        # If smaller than 0, replace entry with 0. Otherwise y=x
        self.output = np.maximum(0,inputs)

# Creating the Softmax activation function
class Activation_Softmax:

    # Forward pass
    def forward(self, inputs):

        # We subtract the largest value of the inputs to avoid an overflow error. This ensures outputs are in a range from some negative value up to 0 (the max)
        # This does not change the final output
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))

        # Normalize them
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities

# Creating the loss class
# Common loss class
class Loss:
    # Calculates the data and regularization losses
    # given model output and ground truth values
    def calculate(self, output, y):
        # Calculate sample losses
        sample_losses = self.forward(output, y)
        # Calculate mean loss
        data_loss = np.mean(sample_losses)
        # Return loss
        return data_loss
# Cross-entropy loss



# Cross-entropy loss
class Loss_CategoricalCrossentropy(Loss):
    # Forward pass
    def forward(self, y_pred, y_true):
        # Number of samples in a batch
        samples = len(y_pred)
        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        # Probabilities for target values -
        # only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
            range(samples),
            y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
            y_pred_clipped*y_true,
            axis=1
            )
        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods

# Lets now create a dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2,3)

# Lets create the output layer (3 input, 3 output)
dense2 = Layer_Dense(3,3)

# Lets create our activation
activation1 = Activation_ReLu()

# Let create out output activation (softmax)
activation2 = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

# Lets now perform a forward pass of the training data through the newly created layer
dense1.forward(X)

# Simply pass output through our activation function
activation1.forward(dense1.output)

# Pass through the second layer
dense2.forward(activation1.output)

# Pass through the second layers activation function
activation2.forward(dense2.output)

print(activation2.output[:5])

loss = loss_function.calculate(activation2.output,y)

print('Loss: ', loss)


# Calculate accuracy from output of activation2 and targets
# calculate values along first axis
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
# Print accuracy
print('acc:', accuracy)