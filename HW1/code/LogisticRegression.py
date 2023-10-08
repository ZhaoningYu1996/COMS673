import numpy as np
import sys

"""This script implements a two-class logistic regression model.
"""

class logistic_regression(object):
	
    def __init__(self, learning_rate, max_iter):
        self.learning_rate = learning_rate
        self.max_iter = max_iter

    def fit_GD(self, X, y):
        """Train perceptron model on data (X,y) with GD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        n_samples, n_features = X.shape

		### YOUR CODE HERE
        self = self.fit_BGD(X, y, n_samples)
		### END YOUR CODE
        return self

    def fit_BGD(self, X, y, batch_size):
        """Train perceptron model on data (X,y) with BGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.
            batch_size: An integer.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        
        weights = np.zeros(n_features)
        self = self.assign_weights(weights)
        # print(self.W)
        for epoch in range(self.max_iter):
            if len(X)==n_samples:
                shuffled_indices = np.arange(len(X))
                # np.random.shuffle(shuffled_indices)
            shuffled_X = X[shuffled_indices]
            shuffled_y = y[shuffled_indices]
            for i in range(int(n_samples/batch_size)):
                sample_X = shuffled_X[i*batch_size:(i+1)*batch_size]
                sample_y = shuffled_y[i*batch_size:(i+1)*batch_size]
                # if i==1:
                #     print(sample_X)
                total_gradient = np.zeros(n_features)
                for j in range(len(sample_X)):
                    gradient = self._gradient(sample_X[j], sample_y[j])
                    total_gradient = total_gradient + gradient
                self.W = self.W - (self.learning_rate*(total_gradient/len(sample_X)))
                # print(self.W)
                if epoch==0 and i==2:
                    first_gradient = total_gradient
                    weight = self.W
                
		### END YOUR CODE
        return self, first_gradient, weight
        # return self

    def fit_SGD(self, X, y):
        """Train perceptron model on data (X,y) with SGD.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
		### YOUR CODE HERE
        self = self.fit_BGD(X, y, 1)
		### END YOUR CODE
        return self

    def _gradient(self, _x, _y):
        """Compute the gradient of cross-entropy with respect to self.W
        for one training sample (_x, _y). This function is used in fit_*.

        Args:
            _x: An array of shape [n_features,].
            _y: An integer. 1 or -1.

        Returns:
            _g: An array of shape [n_features,]. The gradient of
                cross-entropy with respect to self.W.
        """
		### YOUR CODE HERE
        y_pred = 1/(1 + np.exp(-(np.dot(_x, self.W))))
        # print(y_pred)
        gradient = -0.5*(1-2*y_pred+_y)*_x
        # print(gradient)
        return gradient
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

    def predict_proba(self, X):
        """Predict class probabilities for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds_proba: An array of shape [n_samples, 2].
                Only contains floats between [0,1].
        """
		### YOUR CODE HERE
        label_1 = 1/(1 + np.exp(-(np.dot(X, self.W))))
        label_2 = 1 - label_1
        preds_proba = np.vstack((label_2, label_1))
        return preds_proba.T
		### END YOUR CODE


    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
		### YOUR CODE HERE
        preds_proba = self.predict_proba(X)
        preds = np.argmax(preds_proba, axis=1)
        # print(preds.shape)
        preds[preds==0] = -1
        return preds
		### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
		### YOUR CODE HERE
        n_samples, n_features = X.shape
        preds = self.predict(X)
        accuracy = (preds==y).sum()/n_samples
        return accuracy
		### END YOUR CODE
    
    def assign_weights(self, weights):
        self.W = weights
        return self

