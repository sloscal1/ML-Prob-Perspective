import pytest
import numpy as np
import pandas as pd

from chapter.four import QuadraticDiscriminantAnalysis
from chapter.four import LinearDiscriminantAnalysis
from chapter.four import decision_boundary


@pytest.fixture
def height_weight_data():
    return pd.read_csv(
        "cached_data/height_weight.csv",
        header=None,
        names=["gender", "height", "weight"],
        sep=",",
    )


@pytest.fixture
def qda():
    return QuadraticDiscriminantAnalysis()


@pytest.fixture
def lda():
    return LinearDiscriminantAnalysis()


def test_qda_fit(qda, height_weight_data):
    qda.fit(
        height_weight_data.drop(columns=["gender"]).values,
        height_weight_data.gender.values
    )
    assert qda.pi is not None, "Class prior probabilities are set"
    assert qda.mu is not None, "Class means are set"
    assert len(qda.mu) == 2, "Number of means equals the number of classes"
    assert qda.mu[0].shape == (2, ), "Means are the correct shape"
    assert qda.sigma is not None, "Class covariance are set"
    assert len(qda.sigma) == 2, "Number of covariances equals the number of classes"
    assert qda.sigma[0].shape == (2, 2), "Covariances are the correct shape"


def test_qda_predict(qda, height_weight_data):
    qda.fit(
        height_weight_data.drop(columns=["gender"]).values,
        height_weight_data.gender.values
    )
    retval = qda.predict(height_weight_data.drop(columns=["gender"]))
    assert len(retval) == len(height_weight_data), "All elements are predicted"
    assert set(retval).issubset(set(height_weight_data.gender.values)), "All elements are viable class labels"


def test_lda_fit(lda, height_weight_data):
    lda.fit(
        height_weight_data.drop(columns=["gender"]).values,
        height_weight_data.gender.values
    )
    assert lda.pi is not None, "Class prior probabilities are set"
    assert lda.mu is not None, "Class means are set"
    assert len(lda.mu) == 2, "Number of means equals the number of classes"
    assert lda.mu[0].shape == (2, ), "Means are the correct shape"
    assert lda.sigma is not None, "Class covariance is set"
    assert lda.sigma.shape == (2, 2), "Single covariance matrix of correct size"


def test_lda_predict(lda, height_weight_data):
    lda.fit(
        height_weight_data.drop(columns=["gender"]).values,
        height_weight_data.gender.values
    )
    retval = lda.predict(height_weight_data.drop(columns=["gender"]))
    assert len(retval) == len(height_weight_data), "All elements are predicted"
    assert set(retval).issubset(set(height_weight_data.gender.values)), "All elements are viable class labels"


def test_decision_boundary(lda, height_weight_data):
    decision_boundary(
       lda,
       height_weight_data.drop(columns=["gender"]).values,
       height_weight_data.gender.values
    )
    assert False
