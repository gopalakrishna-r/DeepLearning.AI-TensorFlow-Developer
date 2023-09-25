import hashlib
from pathlib import Path

import numpy as np
import requests
import tensorflow as tf

from .HypeParams import HyperParameters

DATA_HUB = {}
DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"


def download(url, folder="../data", sha1_hash=None):
    """Download a file to folder and return the local filepath.

    Defined in :numref:`sec_utils`"""
    if not url.startswith("http"):
        # For back compatability
        url, sha1_hash = DATA_HUB[url]
    Path(folder).mkdir(parents=True, exist_ok=True)
    fname = Path.joinpath(Path(folder), url.split("/")[-1])
    # Check if hit cache
    if fname.exists() and sha1_hash:
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                if data := f.read(1048576):
                    sha1.update(data)
                else:
                    break
        if sha1.hexdigest() == sha1_hash:
            return fname
    # Download
    print(f"Downloading {fname} from {url}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname


class DataModule(HyperParameters):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self, root="../data"):
        self.save_hyperparameters()
        self.is_test = lambda x, y: x % 5 == 0
        self.is_train = lambda x, y: x % 5 != 0

    # def __len__(self):
    #     return self.train_dataloader().unbatach().map(
    #         lambda *x: 1, num_parallel_calls=tf.data.AUTOTUNE
    #     ).reduce(tf.constant(0), lambda x, _: x + 1)

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, indices=slice(0, None)):
        tensors = tuple(a[indices] for a in tensors)
        shuffle_buffer = tensors[0].shape[0] if train else 1
        return (
            tf.data.Dataset.from_tensor_slices(tensors)
            .shuffle(buffer_size=shuffle_buffer)
            .batch(self.batch_size)
        )

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader((self.X, self.y), train, i)


class DatasetModule(HyperParameters):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self):
        self.save_hyperparameters()

    def get_dataloader(self, train):
        raise NotImplementedError

    def train_dataloader(self):
        return self.get_dataloader(train=True)

    def val_dataloader(self):
        return self.get_dataloader(train=False)

    def get_tensorloader(self, tensors, train, split: int):
        if train:
            return tensors.window(split, split + 1).flat_map(lambda _: _)
        else:
            return tensors.skip(split).window(1, split + 1).flat_map(lambda _: _)
