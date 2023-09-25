import tensorflow as tf
from common.Module import Module
from common.linear.SGD import SGD


class LinearRegressionScratch(Module):  # @save
    """The linear regression model implemented from scratch."""

    def __init__(self, num_inputs, lr, sigma=0.01):
        super().__init__()
        self.save_hyperparameters()
        w = tf.random.normal((num_inputs, 1), mean=0, stddev=0.01)
        b = tf.zeros(1)
        self.w = tf.Variable(w, trainable=True)
        self.b = tf.Variable(b, trainable=True)

    def forward(self, X):
        return tf.matmul(X, self.w) + self.b

    def loss(self, y_hat, y):
        l = (y_hat - y) ** 2 / 2
        return tf.reduce_mean(l)

    def configure_optimizers(self):
        return SGD(self.lr)
