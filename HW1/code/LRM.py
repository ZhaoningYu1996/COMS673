import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression_multiclass(object):
	
    def __init__(self, learning_rate, max_iter, k):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.k = k 
    
    def one_hot(self, labels, c):
        y_hot = np.zeros((len(labels), c))
        y_hot[np.arange(len(labels)), labels.astype(int)] = 1
        return y_hot
        
    def fit_BGD(self, X, labels, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,].  Only contains 0,..,k-1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.

        Hint: the labels should be converted to one-hot vectors, for example: 1----> [0,1,0]; 2---->[0,0,1].
        """

		### YOUR CODE HERE
        n_samples, n_features = X.shape
        weights = np.zeros((n_features, self.k))
        self = self.assign_weights(weights)
        # print(self.W)
        for epoch in range(self.max_iter):
            shuffled_indices = np.arange(len(X))
            # np.random.shuffle(shuffled_indices)
            shuffled_X = X[shuffled_indices]
            shuffled_labels = labels[shuffled_indices]
            for i in range(int(n_samples/batch_size)):
                sample_X = shuffled_X[i*batch_size:(i+1)*batch_size]
                sample_labels = shuffled_labels[i*batch_size:(i+1)*batch_size]
                # if i==1:
                #     print(sample_X)
                total_gradient = np.zeros((n_features, self.k))
                sample_labels_hot = self.one_hot(sample_labels, self.k)
                for j in range(len(sample_X)):
                    gradient = self._gradient(sample_X[j].T, sample_labels_hot[j])
                    total_gradient = total_gradient + gradient
                # print(total_gradient)
                self.W = self.W - (self.learning_rate*(total_gradient/len(sample_X)))
                # print(self.W)
                if epoch==0 and i==2:
                    first_gradient = total_gradient
                    weight = self.W      
        return self, first_gradient, weight
        # return self
		### END YOUR CODE
    

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: One_hot vector. 

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        y_pred = self.softmax(_x.T @ self.W)
        _g = np.outer(_x, (y_pred - _y))
        return _g
		### END YOUR CODE
    
    def softmax(self, x):
        """Compute softmax values for each sets of scores in x."""
        ### You must implement softmax by youself, otherwise you will not get credits for this part.

		### YOUR CODE HERE
        # print(x)
        x = x.astype(np.float128)
        exp = np.exp(x-np.max(x))
        softmax = np.zeros(self.k)
        for i in range(len(x)):
            softmax[i] = exp[i] / np.sum(exp)
        return np.exp(x) / np.sum(np.exp(x))

		### END YOUR CODE
    
    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 0,..,k-1.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = np.zeros(n_samples)
        a = X @ self.W
        for i in range(n_samples):
            preds[i] = np.argmax(self.softmax(a[i]))
        return preds
		### END YOUR CODE


    def score(self, X, labels):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            labels: An array of shape [n_samples,]. Only contains 0,..,k-1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. labels.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        accuracy = (preds==labels).sum()/n_samples
        return accuracy
		### END YOUR CODE

    def assign_weights(self, weights):
        self.W = weights
        return self