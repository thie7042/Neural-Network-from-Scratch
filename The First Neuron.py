import numpy as np

###################################
#    Coding our first neurons     #
###################################

# Let''s assume a layer with 3 inputs and 4 neurons

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