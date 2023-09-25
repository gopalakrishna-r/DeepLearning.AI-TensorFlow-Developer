import tensorflow as tf

from common.Module import Module


class LinearRegression(Module):
    """The linear regression model implemented with high-level APIs.

    Defined in :numref:`sec_linear_concise`"""

    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        initializer = tf.initializers.RandomNormal(stddev=0.01)
        self.net = tf.keras.models.Sequential(
            [tf.keras.layers.Dense(1, kernel_initializer=initializer)]
        )

    def forward(self, X):
        """Defined in :numref:`sec_linear_concise`"""
        print(f"line reg {X}")
        return self.net(X)

    def loss(self, y_hat, y):
        """Defined in :numref:`sec_linear_concise`"""
        fn = tf.keras.losses.MeanSquaredError()
        return fn(y, y_hat)

    def configure_optimizers(self):
        """Defined in :numref:`sec_linear_concise`"""
        return tf.keras.optimizers.SGD(self.lr)

    def get_w_b(self):
        """Defined in :numref:`sec_linear_concise`"""
        return self.get_weights()[0], self.get_weights()[1]
