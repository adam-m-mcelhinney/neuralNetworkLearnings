# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 13:39:47 2016

@author: adam.mcelhinney
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 09:49:55 2016

@author: adam.mcelhinney


Example Neural Network
"""

# Package imports
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import sklearn.datasets
import sklearn.linear_model
import matplotlib
from scipy.optimize import brute


matplotlib.rcParams['figure.figsize'] = (10.0, 8.0)

# Generate a dataset and plot it
np.random.seed(0)
X, y = sklearn.datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)

nn_input_dim = 2
nn_output_dim = 3


# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
alpha = .01
k = 0
reg_lambda = 0.01 # regularization strength


# Helper function to evaluate the total loss on the dataset
# Note that they are using a cross-entropy loss function
# TODO: Refactor this to test different loss functions
def forward_prop(W1, b1, W2, b2, x):
    z1 = x.dot(W1) + b1
    a1 = np.tanh(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return (probs, a1)



def calculate_loss(model, x, y):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # DONE: Abstract out the forward propogation to its own function
    # Forward propagation to calculate our predictions
    probs = forward_prop(W1, b1, W2, b2, x)[0]
    num_examples = len(x) # training set size
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, x):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation
    probs, a1 = forward_prop(W1, b1, W2, b2, x)
    return np.argmax(probs, axis=1)


def learningRate(i, alpha = .01, k = 0):
    '''Decay the learning rate using the form
    epsilon = alpha * exp(-k * i)
    where
    alpha: hyper parameter
    k: hyper parameter
    i: iteration number
    Note this defaults to a fixed .01 decay rate
    '''
    epsilon = alpha * np.exp(-k * i)
    return epsilon




# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(inputVals):


    print_loss=False
    returnVal = 'loss'
    #x, y = X, y
    nn_hdim, num_passes,nn_input_dim, nn_output_dim, alpha, k = inputVals
    num_examples = len(x) # training set size

    # Initialize the parameters to random values. We need to learn these.
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in xrange(0, num_passes):

        # Forward propagation
        probs, a1 = forward_prop(W1, b1, W2, b2, x)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(x.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        epsilon = learningRate(i, alpha = alpha, k = k)
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        loss = calculate_loss(model, x, y)
        if print_loss and i % 1000 == 0:
          print "Loss after iteration %i: %f" %(i, loss)
    if returnVal == 'loss':
        return loss
    else:
        return model

def plot_decision_boundary(pred_func):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)


# Build a model with a 3-dimensional hidden layer
params = (X, y)
nn_hdim = 3
num_passes = 20000
nn_input_dim = 2
nn_output_dim = 2
alpha = .01
k = 0
inputVals = (nn_hdim, num_passes,nn_input_dim, nn_output_dim, alpha, k)
model = build_model(inputVals)


range = (slice(2,5,1), slice(2000, 20000, 1000), slice(2,3, 1), slice(2, 3, 1), slice(.01, .01, 1), slice(1,1,1))
brute(model, ranges = range2)



