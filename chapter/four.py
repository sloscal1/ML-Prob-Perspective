import math

import numpy as np
import pandas as pd
import holoviews as hv


class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.pi = None
        self.mu = None
        self.sigma = None
        self.pos_labels = None
        self.probs = None

    def fit(self, data, label):
        self.pos_labels, counts = np.unique(label, return_counts=True)
        # Each of the pi's are the MLE of the class probabilities
        self.pi = counts/len(label)
        # Each of the mu's are the MLE of the class means
        self.mu = [np.mean(data[np.nonzero(label == cl)], axis=0) for cl in self.pos_labels]
        # Each of the sigma's are the MLE of the class STD
        self.sigma = []
        for cl, n_c, mu in list(zip(self.pos_labels, counts, self.mu)):
            dist = (data[np.nonzero(label == cl)] - mu).transpose()
            self.sigma.append(dist @ dist.transpose()/n_c)

    def predict(self, data):
        # Need to use the log-sum-exp trick (pg 88)
        # p(y=c|x) = p(y=c)p(x|y=c)/\sum_c' p(x|y=c')p(y=c)
        # = log(p(y=c)) + log(p(x|y=c)) - [log(\sum_c' p(x|y=c')p(y=c))]
        # = ... - log\sum_c' e^{log(p(x|y=c)p(y=c))}
        # = ... - log\sum_c' e^{log(p(x|y=c)) + log(p(y=c)}
        # = ... - log e^{max_c'}e^{0 + c_i-max_c' + c_j-max_c'}
        # = ... - log e^{max_c'} + log e^{0 + ... + c_j-max_c'}
        # = ... - max_c' + log\sum e^{0 + ... + c_j-max_c'}
        if self.pi is None:
            raise ValueError("Fit the model first!")
        if isinstance(data, pd.DataFrame):
            data = data.values
        # Get the log probabilities of each class for each sample according to the MLE
        num = []
        for pi, mu, sigma in list(zip(self.pi, self.mu, self.sigma)):
            vary = data - mu
            num.append(
                # These terms could be factored out of all future predictions
                # as well as sigma^-1
                np.log(pi) + -0.5*np.log(np.linalg.det(2*math.pi*sigma))
                -0.5 * np.diagonal(vary @ np.linalg.inv(sigma) @ vary.T)
            )
        num = np.vstack(num)
        b = num.max(axis=0)
        denom = b + np.log(np.sum(np.exp(num - b), axis=0))
        # Prob: np.exp(num - denom)
        self.probs = (num - denom)
        return [self.pos_labels[i] for i in self.probs.argmax(axis=0)]


class LinearDiscriminantAnalysis:
    def __init__(self):
        self.pi = None
        self.mu = None
        self.sigma = None
        self.pos_labels = None
        self.probs = None

    def fit(self, data, label):
        self.pos_labels, counts = np.unique(label, return_counts=True)
        # Each of the pi's are the MLE of the class probabilities
        self.pi = counts/len(label)
        # Each of the mu's are the MLE of the class means
        self.mu = [np.mean(data[np.nonzero(label == cl)], axis=0) for cl in self.pos_labels]
        # Each of the sigma's are the MLE of the class STD
        self.sigma = []
        for cl, n_c, mu in list(zip(self.pos_labels, counts, self.mu)):
            dist = (data[np.nonzero(label == cl)] - mu).transpose()
            self.sigma.append(dist @ dist.transpose()/n_c)
        self.sigma = np.dot(np.dstack(self.sigma), self.pi)

    def predict(self, data):
        if self.pi is None:
            raise ValueError("Fit the model first!")
        if isinstance(data, pd.DataFrame):
            data = data.values
        # Get the log probabilities of each class for each sample according to the MLE
        num = []
        gamma = np.linalg.inv(self.sigma)
        for pi, mu in list(zip(self.pi, self.mu)):
            # These terms could be factored out of all future predictions
            # as well as sigma^-1
            num.append(np.log(pi) - 0.5*(mu.T @ gamma @ mu) + (data @ gamma @ mu))
        num = np.vstack(num)
        self.probs = num
        return [self.pos_labels[i] for i in num.argmax(axis=0)]


def decision_boundary(model, data, target):
    hv.extension("bokeh")
    # Get the classes
    classes = np.unique(target)
    # Get a grid
    bounds = [data.min(axis=0), data.max(axis=0)]
    points = []
    for dim in range(data.shape[1]):
        points.append(np.linspace(bounds[0][dim], bounds[1][dim], 25))
    xx, yy = np.meshgrid(np.linspace(bounds[0][0], bounds[1][1], 25),
                       np.linspace(bounds[0][1], bounds[1][1], 25),
                       indexing="xy")
    grid = np.vstack([xx.ravel(), yy.ravel()]).T
    # Predict those points
    model.fit(data, target)
    preds = np.array(model.predict(grid))
    grid = np.hstack([grid, preds[:, None]])
    # Draw them!
    img = hv.Image(grid, bounds=list(bounds[0])+list(bounds[1]))
    img
    # Draw the actual points


