import math

import numpy as np
import pandas as pd


class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.pi = None
        self.mu = None
        self.sigma = None
        self.pos_labels = None

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
        return [self.pos_labels[i] for i in (num - denom).argmax(axis=0)]


