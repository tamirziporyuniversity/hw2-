#!/usr/bin/env python

import numpy as np
import random

from softmax import softmax
from sigmoid import sigmoid, sigmoid_grad
from gradcheck import gradcheck_naive


def forward(data, label, params, dimensions):
    """
    runs a forward pass and returns the probability of the correct word for eval.
    label here is an integer for the index of the label.
    This function is used for model evaluation.
    """
    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs + Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    # Compute the probability
    ### YOUR CODE HERE: forward propagation
    
    #hidden layer
    z0 = data
    v1 = np.dot(z0, W1)+b1
    z1 = sigmoid(v1)
   
    v2 = np.dot(z1, W2)+b2

    output = softmax(v2)
    output = output.ravel()

    #print(output[0][label])

    return output[label]


    raise NotImplementedError
    ### END YOUR CODE



def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """

    """
    our reference
    https://datascience.stackexchange.com/questions/20139/gradients-for-bias-terms-in-backpropagation
    """

    # Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

    ### YOUR CODE HERE: forward propagation
    #raise NotImplementedError
    z0 = data
    v1 = np.dot(z0, W1)+b1
    z1 = sigmoid(v1)
   
    v2 = np.dot(z1, W2)+b2

    output = softmax(v2)


    ### END YOUR CODE
    cost = -np.sum(labels * np.log(output)) 
       
    ### YOUR CODE HERE: backward propagation
    #raise NotImplementedError
    gradW1 = np.zeros(W1.shape)
    gradW2 = np.zeros(W2.shape)
    gradb1 = np.zeros(b1.shape)
    gradb2 = np.zeros(b2.shape)

 
    delta2 = output-labels
    #when I write gradW2 = np.dot(delta2.T, z1) it's not work. ask why
    gradW2 = np.dot(z1.T, delta2)
    gradb2 = delta2
    gradb2 = delta2.sum(axis = 0)
    
    #delta1 = np.dot(sigmoid_grad(v1), W2) * delta2
    delta1 =  sigmoid_grad(z1) * np.dot(delta2, W2.T)
    gradW1 = np.dot(z0.T, delta1)
    gradb1 = np.sum(delta1, axis=0)

    ### END YOUR CODE

    # Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print("Running sanity check...")

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in range(N):
        labels[i, random.randint(0, dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
                    forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q1c_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print("Running your sanity checks...")
    ### YOUR OPTIONAL CODE HERE
    pass
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
