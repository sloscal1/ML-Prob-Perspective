import math
import shutil
import time
from pathlib import Path
from gzip import GzipFile

import urllib3
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

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

    def predict(self, X, batch_size=2048):
        # Split up the computation because the matrix gets too
        # big for memory if done at once.
        # The appropriate batch size should be figured out per machine
        # and data dimensionality.
        preds = []
        batches = int(math.ceil(X.shape[0]/batch_size))
        for batch in range(batches):
            start = batch*batch_size
            end = min((batch+1)*batch_size, X.shape[0])

            # Find the distance from every point to the training data
            # This could be done using partial evaluation to speed this up.
            dists = cdist(self.X, X[start:end])
            # Each row is the distance from all test points to 1 training sample
            # I need the min k values in each column
            partitioned = np.argpartition(dists, self.k, axis=0)
            nearest = np.squeeze(partitioned[:self.k, :])
            # Get the labels corresponding to the min positions
            labels = np.repeat(self.y.values, self.k, axis=0)
            if self.k > 1:
                next_preds = np.squeeze(mode(labels[nearest], axis=0).mode)
            next_preds = np.squeeze(labels[nearest])
            preds = preds + next_preds.tolist()
        return preds

class FLANN_KNN(object):
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
        flann = FLANN()
        nearest, _ = flann.nn(
                self.X.values,
                X.values,
                self.k,
        )

        # Get the labels corresponding to the min positions
        labels = np.repeat(self.y.values, self.k, axis=0)
        if self.k > 1:
            return np.squeeze(mode(labels[nearest], axis=1).mode)
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
    error_rate = accuracy_score(preds, test_df[:1000].target)
    print(f"Error rate of first 1000: {100-error_rate*100:0.2f}%")
    
    preds = kNN.predict(test_df.drop(columns=["target"]))
    error_rate = accuracy_score(preds, test_df.target)
    print(f"Overall error rate: {100-error_rate*100:0.2f}%")

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

    In the end, the speedup is quite drastic. As the sample size doubles
    from 500 to 1000 to 2000 there is almost no change in speed for the FLANN
    version; however, the time increases substantially for the Linear version.

    .. _yourself: https://github.com/primetang/pyflann/issues/1
    """
    dtypes = dict([(num, np.uint8) for num in range(28*28)])
    train_df = pd.read_csv("cached_data/mnist_train.csv", sep=",", dtype=dtypes)
    test_df = pd.read_csv("cached_data/mnist_test.csv", sep=",", dtype=dtypes)

    results = []
    for samps in [500, 1000, 2000]:
        for (alg, name) in [(FLANN_KNN(1), "flann"), (LinearKNN(1), "linear")]:
            start = time.time()
            alg.fit(train_df.drop(columns=["target"]), train_df.target)
            preds = alg.predict(test_df[:samps].drop(columns=["target"]))
            end = time.time()
            error_rate = 100-accuracy_score(preds, test_df[:samps].target)*100
            results.append([name, samps, end-start, error_rate])
    results_df = pd.DataFrame(
        results,
        columns=[
            'algorithm',
            'n',
            'time',
            'error_rate'
        ],
    )
    print(results_df)


def question_3():
    dtypes = dict([(num, np.uint8) for num in range(28*28)])
    train_df = pd.read_csv("cached_data/mnist_train.csv", sep=",", dtype=dtypes)
    test_df = pd.read_csv("cached_data/mnist_test.csv", sep=",", dtype=dtypes)

    data = []
    kNN = FLANN_KNN(1)
    kNN.fit(train_df.drop(columns=["target"]), train_df.target)
    for k in range(1, 11):
        kNN.k = k
        train_preds = kNN.predict(train_df.drop(columns=["target"]))
        train_acc = accuracy_score(train_preds, train_df.target)
        test_preds = kNN.predict(test_df.drop(columns=["target"]))
        test_acc = accuracy_score(test_preds, test_df.target)
        data.append([k, train_acc, test_acc])
    rates = pd.DataFrame(data, columns=["k", "train_acc", "test_acc"])
    
    plt.title("kNN Accuracy Rates for Given k")
    plt.plot("k", "train_acc", data=rates)
    plt.plot("k", "test_acc", data=rates)
    plt.show()
        


if __name__ == "__main__":
    question_3()
