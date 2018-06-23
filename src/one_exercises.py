import shutil
from pathlib import Path
from gzip import GzipFile

import urllib3
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from pyflann import *

# This data comes from: http://yann.lecun.com/exdb/mnist/
def download_if_needed(url, local):
    if not Path(local).exists():
        Path.mkdir(Path(local).parent, parents=True, exist_ok=True)
        pool = urllib3.PoolManager()
        with pool.request("GET", url, preload_content=False) as resp:
            with open(local, "wb") as out_file:
                shutil.copyfileobj(resp, out_file)
    return None

def read_mnist_img(local):
    if not Path(local+".csv").exists():
        with GzipFile(local, "rb") as unzipped:
            magic = int.from_bytes(unzipped.read(4), byteorder='big')
            num_images = int.from_bytes(unzipped.read(4), byteorder='big')
            num_rows = int.from_bytes(unzipped.read(4), byteorder='big')
            num_cols = int.from_bytes(unzipped.read(4), byteorder='big')
            data = []
            for _ in range(num_images):
                img = []
                for _ in range(num_rows):
                    row = []
                    for _ in range(num_cols):
                        row.append(int.from_bytes(unzipped.read(1), byteorder='big', signed=False))
                    img.extend(row)
                data.append(img)
            df = pd.DataFrame(data)
            df.to_csv(local+".csv", sep=",", index=False)
    else:
        df = pd.read_csv(local+".csv", sep=",")
    return df

def read_mnist_labels(local):
    if not Path(local+".csv").exists():
        with GzipFile(local, "rb") as unzipped:
            magic = int.from_bytes(unzipped.read(4), byteorder='big')
            num_items = int.from_bytes(unzipped.read(4), byteorder='big')
            labels = []
            for _ in range(num_items):
                labels.append(int.from_bytes(unzipped.read(1), byteorder='big', signed=False))
            df = pd.DataFrame(labels, columns=["target"])
            df.to_csv(local+".csv", sep=",", index=False)
    else:
        df = pd.read_csv(local+".csv", sep=",")
    return df

class LinearKNN(object):
    def __init__(self, k):
        if not (k % 2):
            raise ValueError("k must be odd to break ties")
        self.k = k
        self.X = None
        self.y = None

    def fit(self, X, y):
        self.X = X
        self.y = y

    def predict(self, X):
        # Find the distance from every point to the training data
        # This could be done using partial evaluation to speed this up.
        dists = cdist(self.X, X)
        # Each row is the distance from all test points to 1 training sample
        # I need the min k values in each column
        partitioned = np.argpartition(dists, self.k, axis=0)
        nearest = np.squeeze(partitioned[:self.k, :])
        # Get the labels corresponding to the min positions
        labels = np.repeat(self.y.values, self.k, axis=0)
        if self.k > 1:
            return np.squeeze(mode(labels[nearest], axis=0).mode)
        return np.squeeze(labels[nearest])

def question_1():
    if not (
            Path("cached_data/mnist_train.csv").exists()
            and Path("cached_data/mnist_test.csv").exists()):
        # Get the MNIST data:
        download_if_needed(
                "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                "cached_data/mnist_train.gz"
        )
        download_if_needed(
                "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
                "cached_data/mnist_train_labels.gz"
        )
        download_if_needed(
                "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "cached_data/mnist_test.gz"
        )
        download_if_needed(
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
                "cached_data/mnist_test_labels.gz"
        )
        train_df = read_mnist_img("cached_data/mnist_train.gz") 
        test_df = read_mnist_img("cached_data/mnist_test.gz")
        train_labels = read_mnist_labels("cached_data/mnist_train_labels.gz")
        test_labels = read_mnist_labels("cached_data/mnist_test_labels.gz")
        train_df = pd.concat([train_df, train_labels], axis="columns")
        test_df = pd.concat([test_df, test_labels], axis="columns")
        train_df.to_csv("cached_data/mnist_train.csv", sep=",", index=False)
        test_df.to_csv("cached_data/mnist_test.csv", sep=",", index=False)
    else:
        dtypes = dict([(num, np.uint8) for num in range(28*28)])
        train_df = pd.read_csv("cached_data/mnist_train.csv", sep=",", dtype=dtypes)
        test_df = pd.read_csv("cached_data/mnist_test.csv", sep=",", dtype=dtypes)

    kNN = LinearKNN(1)
    kNN.fit(train_df.drop(columns=["target"]), train_df.target)
    preds = kNN.predict(test_df[:1000].drop(columns=["target"]))
    error_rate = [accuracy_score(preds, test_df[:1000].target)]
    print(f"Error rate of first 1000: {100-error_rate[-1]*100:0.2f}%")
    
    # Couldn't do all 10000 on my machine in one go.
    for end_val in range(2000, 10001, 1000):
        preds = kNN.predict(test_df[end_val-1000:end_val].drop(columns=["target"]))
        error_rate.append(accuracy_score(preds, test_df[end_val-1000:end_val].target))
        print(f"Error rate of samples {end_val-1000} to {end_val}: {100-error_rate[-1]*100:0.2f}%")
    print(f"Overall error rate: {100-np.mean(error_rate)*100:0.2f}%")

    # Doing the shuffling part:
    np.random.seed = 1337
    idxs = list(range(train_df.shape[1]-1))
    np.random.shuffle(idxs)
    idxs.append(train_df.shape[1]-1) # Keep target at the end
    shuff_train_df = train_df.copy()
    shuff_train_df.columns = train_df.columns[idxs]
    shuff_test_df = test_df.copy()
    shuff_test_df.columns = test_df.columns[idxs]
    kNN.fit(shuff_train_df.drop(columns=["target"]), shuff_train_df.target)
    preds = kNN.predict(shuff_test_df[:1000].drop(columns=["target"]))
    error_rate = [accuracy_score(preds, shuff_test_df[:1000].target)]
    print(f"Error rate of first 1000 shuffled columns: {100-error_rate[-1]*100:0.2f}%")
    

def question_2():
    """
    This code has some external dependencies. Namely, it uses FLANN and the
    python bindings for it. Unfortunately, the maintainer of those bindings
    hasn't fixed some compatibility issues so a pull request needs to be used
    to allow it to work with python 3. Or, you can fix the few compatibillity
    errors yourself_ .

    ```
    git clone https://github.com/primetang/pyflann.git
    cd pyflann
    git fetch origin pull/7/head:python36
    git checkout python36
    python setup.py install
    ```

    .. _yourself: https://github.com/primetang/pyflann/issues/1
    """

    dtypes = dict([(num, np.uint8) for num in range(28*28)])
    train_df = pd.read_csv("cached_data/mnist_train.csv", sep=",", dtype=dtypes)
    test_df = pd.read_csv("cached_data/mnist_test.csv", sep=",", dtype=dtypes)

    flann = FLANN()
    result, dists = flann.nn(
            train_df.drop(columns=["target"]).values,
            test_df.drop(columns=["target"]).values,
            1
    )
    print(result)


if __name__ == "__main__":
    question_2()
