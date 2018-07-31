import numpy as np


class QuadraticDiscriminantAnalysis:
    def __init__(self):
        self.pi = None
        self.mu = None
        self.sigma = None

    def fit(self, data, label):
        pos_labels, counts = np.unique(label, return_counts=True)
        # Each of the pi's are the MLE of the class probabilities
        self.pi = counts/len(label)
        # Each of the mu's are the MLE of the class means
        self.mu = [np.mean(data[np.nonzero(label == cl)], axis=0) for cl in pos_labels]
        # Each of the sigma's are the MLE of the class STD
        self.sigma = []
        for cl, n_c, mu in list(zip(pos_labels, counts, self.mu)):
            dist = (data[np.nonzero(label == cl)] - mu).transpose()
            self.sigma.append((dist @ dist.transpose())/n_c)
