import keras
import tensorflow as tf
from common.DataModule import DataModule

from common.linear.LinearRegression import LinearRegression


class WeightDecay(LinearRegression):
    def __init__(self, wd, lr):
        super().__init__(lr)
        self.save_hyperparameters()
        self.net = keras.layers.Dense(
            1,
            kernel_regularizer=keras.regularizers.l2(wd),
            kernel_initializer=keras.initializers.RandomNormal(0, 0.01),
        )

    def loss(self, y_hat, y):
        return super().loss(y_hat, y) + self.net.losses


def l2_penalty(w):
    return tf.reduce_sum(w**2) / 2


class Data(DataModule):
    def __init__(self, num_train, num_val, num_inputs, batch_size):
        self.save_hyperparameters()
        n = num_train + num_val
        self.X = tf.random.normal((n, num_inputs))
        noise = tf.random.normal((n, 1)) * 0.01
        w, b = tf.ones((num_inputs, 1)) * 0.01, 0.05
        self.y = tf.matmul(self.X, w) + b + noise

    def get_dataloader(self, train):
        i = slice(0, self.num_train) if train else slice(self.num_train, None)
        return self.get_tensorloader([self.X, self.y], train, i)
