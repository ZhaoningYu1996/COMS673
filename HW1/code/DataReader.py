import numpy as np

"""This script implements the functions for reading data.
"""

def load_data(filename):
    """Load a given txt file.

    Args:
        filename: A string.

    Returns:
        raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
        
    """
    data= np.load(filename)
    x= data['x']
    y= data['y']
    return x, y

def train_valid_split(raw_data, labels, split_index):
	"""Split the original training data into a new training dataset
	and a validation dataset.
	n_samples = n_train_samples + n_valid_samples

	Args:
		raw_data: An array of shape [n_samples, 256].
        labels : An array of shape [n_samples,].
		split_index: An integer.

	"""
	return raw_data[:split_index], raw_data[split_index:], labels[:split_index], labels[split_index:]

def prepare_X(raw_X):
    """Extract features from raw_X as required.

    Args:
        raw_X: An array of shape [n_samples, 256].

    Returns:
        X: An array of shape [n_samples, n_features].
    """
    raw_image = raw_X.reshape((-1, 16, 16))
    n_samples = raw_X.shape[0]
    n_features = 3
    width = 16
    n_pixels = 256
    X = np.zeros((n_samples, n_features))
	# Feature 1: Measure of Symmetry
	### YOUR CODE HERE
    # print(raw_image.shape)
    for n in range(n_samples):
        F_s = 0
        for i in range(16):
            for j in range(16):
                F_s = F_s + abs(raw_image[n, i, j] - raw_image[n, i, width-j-1])
        F_s = -F_s/n_pixels
        X[n, 1] = F_s
	### END YOUR CODE

	# Feature 2: Measure of Intensity
	### YOUR CODE HERE
    for n in range(n_samples):
        F_i = 0
        for i in range(16):
            for j in range(16):
                F_i += raw_image[n, i, j]
        F_i = F_i/n_pixels
        X[n, 2] = F_i
	### END YOUR CODE

	# Feature 3: Bias Term. Always 1.
	### YOUR CODE HERE
    for n in range(n_samples):
        X[n, 0] = 1
	### END YOUR CODE

	# Stack features together in the following order.
	# [Feature 3, Feature 1, Feature 2]
	### YOUR CODE HERE
	
	### END YOUR CODE
    return X

def prepare_y(raw_y):
    """
    Args:
        raw_y: An array of shape [n_samples,].
        
    Returns:
        y: An array of shape [n_samples,].
        idx:return idx for data label 1 and 2.
    """
    y = raw_y
    idx = np.where((raw_y==1) | (raw_y==2))
    y[np.where(raw_y==0)] = 0
    y[np.where(raw_y==1)] = 1
    y[np.where(raw_y==2)] = 2

    return y, idx




