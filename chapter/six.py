"""
This module covers some tests from chapter 6: Frequentist Methods.
"""
import numpy as np


class MajorityClassifier:
    """ Selects the majority label from the training data.

    This classifier only works on random data, with a binary label it's not meant
    to be used for actually problems.

    Attributes:
        prob (float): the probability of the majority class.
    """
    def __init__(self):
        self.prob = None

    def fit(self, data):
        """ Find out the majority class label from the test labels.

        Args:
            data (numpy.array): an array of labels :math:`y \in {0, 1}`.

        Returns:
            None.
        """
        self.prob = np.sum(data)/data.shape[0]
        return None

    def predict(self):
        """ Return the majority class irrespective of input.

        Returns:
            int: 1 or 0.
        """
        return 1 if self.prob > 0.5 else 0


class ProbabilisticClassifier:
    """ Selects a class label by the proportion of labels in the training data.

    Attributes:
        prob (float): the probability of getting 1 the data.
    """
    def __init__(self, seed=None):
        self.prob = None
        np.random.seed(seed)

    def fit(self, data):
        """ Find out the probability of 1's from the test labels.

        Args:
            data (numpy.array): an array of labels :math:`y \in {0, 1}`.

        Returns:
            None.
        """
        self.prob = np.sum(data)/data.shape[0]
        return None

    def predict(self):
        """ Return a stochastic result that is proportional to the number of 1's in the training data.

        Returns:
            int: 1 or 0.
        """
        return 1 if np.random.random() < self.prob else 0


def demo_loocv():
    r""" Pessimism of LOOCV.

    I was thinking about this a little, and I was wondering about this question
    beyond what was stated in 6.1.

    Returns:
        None.
    """
    runs = 1000
    samps = 100000
    training_labels = np.concatenate(
        (
            np.ones((50)),
            np.zeros((50))
        )
    )
    np.random.shuffle(training_labels)
    test_labels = np.copy(training_labels)
    np.random.shuffle(test_labels)


    m_holdout = []
    p_holdout = []
    m_loss = []
    p_loss = []
    for _ in range(runs):
        # here is the expected loss of the classifiers using a holdout sample:
        maj_test = MajorityClassifier()
        maj_test.fit(training_labels)
        maj_preds = np.array([maj_test.predict() for _ in range(test_labels.shape[0])])
        m_holdout.append(np.sum(np.abs(maj_preds - test_labels)))
        prob_test = ProbabilisticClassifier()
        prob_test.fit(training_labels)
        prob_preds = np.array([prob_test.predict() for _ in range(test_labels.shape[0])])
        p_holdout.append(np.sum(np.abs(prob_preds - test_labels)))

        majority_loss = 0
        prob_loss = 0
        for i, label in enumerate(training_labels):
            train = np.delete(training_labels, i)
            cls1 = MajorityClassifier()
            cls1.fit(train)
            majority_loss += abs(cls1.predict() - label)
            cls2 = ProbabilisticClassifier()
            cls2.fit(train)
            prob_loss += abs(cls2.predict() - label)
        m_loss.append(majority_loss)
        p_loss.append(prob_loss)

    print(f"Majority classifier hold-out error: {sum(m_holdout)/test_labels.shape[0]/runs*100:0.2f}")
    print(f"Probabilistic classifier hold-out error: {sum(p_holdout)/test_labels.shape[0]/runs*100:0.2f}")
    print(f"Majority classifier LOOCV: {sum(m_loss)/training_labels.shape[0]/runs*100:0.2f}")
    print(f"Probabilistic classifier LOOCV: {sum(p_loss)/training_labels.shape[0]/runs*100:0.2f}")
    return None


if __name__ == "__main__":
    demo_loocv()
