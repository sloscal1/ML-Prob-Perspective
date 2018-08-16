import numpy as np
import pandas as pd

from chapter.four import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis


def question_17():
    """ Misclassification error rate of QDA and LDA.

    Looks like they're both the same for the Height and Weight data: 11.9%.

    Returns:
        None.
    """
    # Get the height weight data
    hw_data = pd.read_csv(
        "cached_data/height_weight.csv",
        header=None,
        names=["gender", "height", "weight"],
        sep=",",
    )
    train = hw_data.drop(columns=["gender"]).values
    lda = LinearDiscriminantAnalysis()
    lda.fit(train, hw_data.gender.values)
    preds = lda.predict(train)
    miss_lda = np.not_equal(preds, hw_data.gender.values).sum()/hw_data.gender.shape[0]
    # Get the missclassification rate on each of LDA and QDA.
    print(f"LDA missclassification rate: {miss_lda*100:0.2f}")
    qda = QuadraticDiscriminantAnalysis()
    qda.fit(train, hw_data.gender.values)
    preds = qda.predict(train)
    miss_qda = np.not_equal(preds, hw_data.gender.values).sum()/hw_data.gender.shape[0]
    print(f"QDA missclassification rate: {miss_qda*100:0.2f}")
    return None


def question_19():
    r""" Stretched LDA.

    This is LDA with the additional parameter that we can multiply the grouped covariance matrix
    by a class-specific value to stretch the Gaussians.

    This change will cause the resulting model to still be quadratic in :math:`x` because we can't
    eliminate all of the squared terms as we did with LDA.

    .. math::
        p(y=c|x,\theta) &= \frac{\pi_c|2\pi\Sigma|^{-\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu_c)^T\Sigma^{-1}(x-\mu_c))}{\Sigma_{c'}\pi_{c'}|2\pi\Sigma|^{-\frac{1}{2}}\exp(-\frac{1}{2}(x-\mu_{c'})^T\Sigma^{-1}(x-\mu_{c'}))}\\
            &= \frac{\pi_ck_c^n\exp(-\frac{1}{2}(x-\mu_c)^T\Sigma^{-1}(x-\mu_c))}{\Sigma_{c'}\pi_{c'}k_{c'}^n\exp(-\frac{1}{2}(x-\mu_{c'})^T\Sigma^{-1}(x-\mu_{c'}))}\\
            &= \frac{\pi_ck_c^n\exp(-\frac{1}{2}k_cx^T\Sigma^{-1}x)\exp(-\frac{1}{2}k_c\mu_c^T\Sigma^{-1}\mu_c+k_c\mu_c^T\Sigma^{-1}x}{\Sigma_{c'}\pi_{c'}k_{c'}^n\exp(-\frac{1}{2}k_{c'}x^T\Sigma^{-1}x)\exp(-\frac{1}{2}k_{c'}\mu_{c'}^T\Sigma^{-1}\mu_{c'}+k_{c'}\mu_{c'}^T\Sigma^{-1}x}\\

    Returns:
        None.
    """
    return None


def question_22():
    r""" QDA with 3 classes.

    Side effect: prints the answer that we're looking for!

    Point [-0.5, 0.5] is classified as 1, while [0.5, 0.5] is classified as 2.

    Interestingly, they're both very close to the decision boundary, with the first point
    almost in the 3rd class, and the second point almost in the 1st class. The results may
    be an artifact due to numerical instability in the code, but visual inspection marks them
    as the class labels given above.

    Returns:
        None.
    """
    qda = QuadraticDiscriminantAnalysis()
    qda.pi = [1 / 3, 1 / 3, 1 / 3]
    qda.mu = [np.array([0, 0]), np.array([1, 1]), np.array([-1, 1])]
    qda.sigma = [
        np.matrix([[0.7, 0], [0, 0.7]]),
        np.matrix([[0.8, 0.2], [0.2, 0.8]]),
        np.matrix([[0.8, 0.2], [0.2, 0.8]]),
    ]
    # Use this instead of the above 4 lines for LDA.
    # qda.sigma = [
    #    np.matrix([[0.766, 1.33], [1.33, 0.766]])
    # ]
    qda.pos_labels = [1, 2, 3]
    print(f"Point (-0.5, 0.5) has label: {qda.predict(np.array([-0.5, 0.5]))}")
    print(f"Point (0.5, 0.5) has label: {qda.predict(np.array([0.5, 0.5]))}")
    return None


def question_23():
    r""" Scalar QDA.

    Side-effect:
        We found the QDA values for each class as desired, and got an 83% probability for a 72 inch
        student to be a male at this university.

        The meatier question is about how to extend this method to the situation if we had multiple
        attributes per person. The clear answer would be to simply run QDA as implemented earlier, but I think
        that's not what the question is really getting at. I think you could run an ensemble of 1D QDA's and
        then take the average of the predicted values for each class across the set of 1D classifiers. This would
        ignore correlations between variables, but would take into account the multiple features.

    Returns:
        None.
    """
    data = np.matrix([
        [67, 79, 71, 68, 67, 60],
        [1, 1, 1, 2, 2, 2],
    ]).T
    qda = QuadraticDiscriminantAnalysis()
    print(list(data[:, 1].flat))
    qda.fit(data[:, 0], list(data[:, 1].flat))
    print(f"Priors: {qda.pi}")
    print(f"Means: {qda.mu}")
    print(f"Sigmas: {qda.sigma}")
    qda.predict(np.array([72]))
    print(f"P(y=m|x=72,\\theta): f{np.exp(qda.probs[0])}")
    return None