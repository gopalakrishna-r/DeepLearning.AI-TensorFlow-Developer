import random
import tensorflow as tf

from common.DataModule import DataModule


class SyntheticRegressionData(DataModule):  # @save
    """Synthetic data for linear regression."""

    def __init__(self, w, b, noise=0.01, num_train=1000, num_val=1000, batch_size=32):
        super().__init__()
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = tf.random.normal((n, w.shape[0]))
        noise = tf.random.normal((n, 1)) * noise
        self.y = tf.matmul(self.X, tf.reshape(w, (-1, 1))) + b + noise

    def __len__(self):
        return len(self.get_dataloader())

    def generate(self, indices):
        for i in range(0, len(indices), self.batch_size):
            j = tf.constant(indices[i : i + self.batch_size])
            yield tf.gather(self.X, j), tf.gather(self.y, j)

    def get_dataloader(self, train):
        if train:
            indices = list(range(self.num_train))
            # The examples are read in random order
            random.shuffle(indices)
        else:
            indices = list(range(self.num_train, self.num_train + self.num_val))

        return list(self.generate(indices))
