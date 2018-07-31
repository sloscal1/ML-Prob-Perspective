import pytest
import pandas as pd

from chapter.four import QuadraticDiscriminantAnalysis


@pytest.fixture
def height_weight_data():
    return pd.read_csv(
        "cached_data/height_weight.csv",
        header=None,
        names=["gender", "height", "weight"],
        sep=","
    )


@pytest.fixture
def qda():
    return QuadraticDiscriminantAnalysis()


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

