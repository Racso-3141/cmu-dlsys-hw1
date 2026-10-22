"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np

import sys

sys.path.append("python/")
import needle as ndl


def parse_mnist(image_filename, label_filename):
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR SOLUTION
    with gzip.open(image_filename, "rb") as f:
        _, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        bytes_per_image = rows * cols
        X = np.zeros((num_images, rows * cols), dtype=np.float32)

        for i in range(num_images):
            image_data = f.read(bytes_per_image)
            if not image_data:
                break
            X[i] = np.frombuffer(image_data, dtype=np.uint8).astype(np.float32)
        X /= 255.0

    with gzip.open(label_filename, "rb") as f:
        _, num_labels = struct.unpack(">II", f.read(8))
        y = np.zeros(num_labels, dtype=np.uint8)
        bytes_per_label = 1
        
        for i in range(num_labels):
            label_data = f.read(bytes_per_label)
            if not label_data:
                break
            y[i] = np.frombuffer(label_data, dtype=np.uint8)[0]
        
    
    return X, y
    ### END YOUR SOLUTION


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    batch_size = Z.shape[0]
    
    exp_Z = ndl.ops.exp(Z)
    exp_Z_sum = ndl.ops.summation(exp_Z, axes=(1,))
    log_exp_Z_sum = ndl.ops.log(exp_Z_sum)
    
    z_y = ndl.ops.summation(ndl.ops.multiply(Z, y_one_hot), axes=(1,))
    
    losses = ndl.ops.add(log_exp_Z_sum, ndl.ops.negate(z_y))
    
    return ndl.ops.divide_scalar(ndl.ops.summation(losses, axes=(0,)), batch_size)
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    num_examples = X.shape[0]

    for batch_start in range(0, num_examples, batch):
        batch_size = min(batch, num_examples - batch_start)
        X_batch = ndl.Tensor(X[batch_start: batch_start + batch_size])
        # Z1 = np.maximum(X_batch@W1, 0)

        # y_batch = y[batch_start: batch_start + batch_size]
        # I_y = np.eye(num_classes)[y_batch]
        # G2 = normalize_exp(Z1@W2) - I_y
        
        # G1 = (Z1 > 0).astype(np.uint8) * (G2@np.transpose(W2))

        # W1 -= lr / batch_size * np.transpose(X_batch)@G1
        # W2 -= lr / batch_size * np.transpose(Z1)@G2
        Z = ndl.ops.relu(X_batch@W1)@W2
        y_batch = y[batch_start: batch_start + batch_size]
        y_batch_one_hot = np.zeros((y_batch.shape[0], Z.shape[-1]))
        y_batch_one_hot[np.arange(y_batch.size), y_batch] = 1
        y_ = ndl.Tensor(y_batch_one_hot)
        loss = softmax_loss(Z, y_)
        loss.backward()
        
        W1 -= lr * W1.grad
        W2 -= lr * W2.grad
    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
