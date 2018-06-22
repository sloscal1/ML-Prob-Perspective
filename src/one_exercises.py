import shutil
from pathlib import Path
from gzip import GzipFile

import urllib3
from scipy.spatial.distance import cdist
from scipy.stats import mode
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

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
        train_df = pd.read_csv("cached_data/mnist_train.csv", sep=",")
        test_df = pd.read_csv("cached_data/mnist_test.csv", sep=",")

    kNN = LinearKNN(1)
    kNN.fit(train_df.drop(["target"], axis="columns"), train_df.target)
    preds = kNN.predict(test_df[:1000].drop(["target"], axis="columns"))
    print(f"Error rate of first 1000: {100-accuracy_score(preds, test_df[:1000].target)*100:0.2f}%")
    
    preds = kNN.predict(test_df.drop(["target"], axis="columns"))
    print(f"Error rate of all data: {100-accuracy_score(preds, test_df.target)*100:0.2f}%")



if __name__ == "__main__":
    question_1()
