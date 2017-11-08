import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim = X.shape
    num_classes = np.max(y) + 1
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]

        loss += -correct_class_score + np.log(np.sum(np.exp(scores)))
        dW += np.exp(scores) * X[i][:, np.newaxis] / np.sum(np.exp(scores))
        dW[:, y[i]] -= X[i]

        # scores -= np.max(scores)
        # scores = np.exp(scores) / np.sum(np.exp(scores))

        # loss += -np.log(scores[y[i]])

    loss /= num_train
    loss += reg * np.sum(W * W)

    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    num_train, dim = X.shape

    scores = X.dot(W)
    scores -= np.max(scores, axis=1, keepdims=True)
    correct_class_scores = scores[np.arange(num_train), y]
    exp_scores = np.exp(scores)
    sum_exp_scores = np.sum(exp_scores, axis=1)
    loss = np.sum(-correct_class_scores + np.log(sum_exp_scores))

    loss /= num_train
    loss += reg * np.sum(W * W)

    mat_correct_scores = np.zeros(scores.shape[::-1])  # transform matrix for correct classes
    mat_correct_scores[y, np.arange(num_train)] = 1
    dW = X.T.dot(-mat_correct_scores.T + exp_scores / sum_exp_scores[:, np.newaxis])
    dW /= num_train
    dW += reg * 2 * W
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW
