####################################
#       Overview of Problem:       #
####################################

# We have 2 datasets. 1 training set and 1 testing
# Our training are labeled as either cat (y=1) or non-cat (y=0)
# Our testing set is also labeled
# Each image is of the shape (number_pixels,number_pixels, 3)
# Each image is square, 3 is the number of channels (RGB)

# We are going to build a very simple image recognition algorithm to classify these images
# Outputs: Either cat (1) or not cat (0)

# The goal of this script is to use as few loops as possible
# Loops are terrible for computational efficiency. We want to Vectorize instead

####################################
#        Importing packages        #
####################################
import sys
import numpy as np
import copy
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage

####################################
#         Loading dataset          #
####################################
def load_dataset():

    with h5py.File('datasets1/train_catvnoncat.h5', "r") as train_dataset:
        train_set_x_orig = np.array(train_dataset["train_set_x"][:])
        train_set_y_orig = np.array(train_dataset["train_set_y"][:])

    with h5py.File('datasets1/test_catvnoncat.h5', "r") as test_dataset:
        test_set_x_orig = np.array(test_dataset["test_set_x"][:])
        test_set_y_orig = np.array(test_dataset["test_set_y"][:])
        classes = np.array(test_dataset["list_classes"][:])

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Let's visualize one of the images and look at what the data says it is.
index = 25
plt.imshow(train_set_x_orig[index])
#plt.show()
print("y = " + str(train_set_y[:, index]) +", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")


####################################
#    Understanding our datasets    #
####################################

# Let's take a look at how many training and testing pictures we have
# Let's also look at the size of the images in pixels

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print("______________________________________________")
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

####################################
#      Reshaping our datasets      #
####################################

# Let's put our pixel data into an appropriate array format
# Shape = (pixels total , number of images)
# Shape = (64 x 64 x 3, number of images)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

print("______________________________________________")
print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# Note: Our image data is currently represented by values ranging between 0 to 255
# Let's standardise our data between 0 to 1

train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255
print("______________________________________________")
print(train_set_x)

####################################
#       General Architecture       #
####################################

# Let's quickly cover the mathematical expressions used
#   z = wb + b
#   y^ = a = sigmoid(z)
#   L(a,y) = - (ylog(a) + (1-y)log(1-a))
# Cost J = 1/m sum(L)

# This will be a very simple neural network, consisting of two layers
# Our input layer (Pixels, number of pictures)
# Our output neuron

####################################
#       Function definitions       #
####################################

# Our chosen activation function is the sigmoid function
def sigmoid(z):
    s = 1 / (1 + np.exp(-z))
    return s

# Let's initialise out parameters (weights and biases) to zero
# Our input dim relates to the number of input variables
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0.0
    return w, b


# Let's conduct forward and backward propagation
# This calculates the activation function, which is used to calculate the cost function
# We can then calculate the gradients dw and db
#     cost -- negative log-likelihood cost for logistic regression
#     dw -- gradient of the loss with respect to w, thus same shape as w
#     db -- gradient of the loss with respect to b, thus same shape as b
def propagate(w, b, X, Y):
    m = X.shape[1]

    A = sigmoid(np.dot(w.T, X) + b)
    cost = 1 / m * np.sum(-(Y * np.log(A) + (1 - Y) * np.log(1 - A)))

    dw = 1 / m * np.dot(X, (A - Y).T)
    db = 1 / m * np.sum(A - Y)

    cost = np.squeeze(np.array(cost))

    # This is a gradient dictionary
    grads = {"dw": dw,
             "db": db}

    return grads, cost


####################################
#           Optimization           #
####################################

# Let's update our parameters (w and b) using gradient descent
#     w -- weights, a numpy array of size (num_px * num_px * 3, 1)
#     b -- bias, a scalar
#     X -- data of shape (num_px * num_px * 3, number of examples)
#     Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
#     num_iterations -- number of iterations of the optimization loop
#     learning_rate -- learning rate of the gradient descent update rule
#     print_cost -- True to print the loss every 100 steps

# This function returns the parameters as a dictionary, the gradients as a dictionary and the costs as a list for plotting

def optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False):

    w = copy.deepcopy(w)
    b = copy.deepcopy(b)

    costs = []

    # Optimization loop
    for i in range(num_iterations):

        # Forward and back propagation for each iteration
        grads, cost = propagate(w, b, X, Y)

        # Retrieve derivatives from grads
        dw = grads["dw"]
        db = grads["db"]

        # update parameters
        w -= learning_rate * dw
        b -= learning_rate * db

        # Record the cost at every 100 training iteration
        if i % 100 == 0:
            costs.append(cost)

            # Print the cost every 100 training iterations
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))

    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs


####################################
#           Prediction             #
####################################

# We now want to predict whether the label is 0 or 1 using learned logistic regression parameters (w, b)
# This returns an array containing all predictions for X
def predict(w, b, X):

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # Compute vector "A" predicting the probabilities of a cat being present in the picture
    # Calculate activation function
    A = sigmoid(np.dot(w.T, X) + b)

    # Make the call by rounding up or down
    for i in range(A.shape[1]):

        # Convert probabilities A[0,i] to actual predictions p[0,i]
        if A[0, i] > 0.5:
            Y_prediction[0, i] = 1
        else:
            Y_prediction[0, i] = 0

    return Y_prediction


###############################################
#           Putting it all together           #
###############################################

# Let's now build the model by calling all functions as needed
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    w, b = initialize_with_zeros(X_train.shape[0])
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)
    w = params["w"]
    b = params["b"]
    Y_prediction_train = predict(w, b, X_train)
    Y_prediction_test = predict(w, b, X_test)


    if print_cost:
        print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
        print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}

    return d

#############################################
#           Running the algorithm           #
#############################################
logistic_regression_model = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)




# Example of a picture that was incorrectly classified.
index = 5
plt.imshow(test_set_x[:, index].reshape((num_px, num_px, 3)))
plt.show()
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[int(logistic_regression_model['Y_prediction_test'][0,index])].decode("utf-8") +  "\" picture.")

# Plot learning curve (with costs)
costs = np.squeeze(logistic_regression_model['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(logistic_regression_model["learning_rate"]))
plt.show()
# Interpretation: You can see the cost decreasing. It shows that the parameters are being learned.
# However, you see that you could train the model even more on the training set.
# Try to increase the number of iterations in the cell above and rerun the cells.
# You might see that the training set accuracy goes up, but the test set accuracy goes down.
# This is called overfitting.


########################################################
#           Testing different Learning Rates           #
########################################################
# In order for Gradient Descent to work you must choose the learning rate wisely. The learning rate determines how rapidly we update the parameters.
# If the learning rate is too large we may "overshoot" the optimal value.
# Similarly, if it is too small we will need too many iterations to converge to the best values.
# That's why it is crucial to use a well-tuned learning rate.

learning_rates = [0.01, 0.001, 0.0001]
models = {}

for lr in learning_rates:
    print ("Training a model with learning rate: " + str(lr))
    models[str(lr)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=lr, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for lr in learning_rates:
    plt.plot(np.squeeze(models[str(lr)]["costs"]), label=str(models[str(lr)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations (hundreds)')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()


# Interpretation:
#
# Different learning rates give different costs and thus different predictions results.
# If the learning rate is too large (0.01), the cost may oscillate up and down. It may even diverge (though in this example, using 0.01 still eventually ends up at a good value for the cost).
# A lower cost doesn't mean a better model. You have to check if there is possibly overfitting. It happens when the training accuracy is a lot higher than the test accuracy.
# In deep learning, it is recommend that we:
#       Choose the learning rate that better minimizes the cost function.
#       If the model overfits, use other techniques to reduce overfitting. (i.e regularization)



#########################################
#           Using a new Image           #
#########################################

# Lets try a random image from the internet
my_image = "new_image.jpg"

# We preprocess the image to fit the algorithm.
fname = "images/" + my_image
image = np.array(Image.open(fname).resize((num_px, num_px)))
plt.imshow(image)
plt.show()
image = image / 255.
image = image.reshape((1, num_px * num_px * 3)).T
my_predicted_image = predict(logistic_regression_model["w"], logistic_regression_model["b"], image)

print("y = " + str(np.squeeze(my_predicted_image)) + ", this algorithm predicts a \"" + classes[int(np.squeeze(my_predicted_image)),].decode("utf-8") +  "\" picture.")

# Note: This script isn't very sophisticated and our data is fairly limited.
# However, as a demo it suits its purpose!