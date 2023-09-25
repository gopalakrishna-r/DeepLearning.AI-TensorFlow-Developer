import keras
import tensorflow as tf

from ..Module import Module


class Classifier(Module):
    """Defined in :numref:`sec_classification`"""

    def validation_step(self, batch):
        y_hat = self(*batch[:-1])
        self.plot("loss", self.loss(y_hat, batch[-1]), train=False)
        self.plot("acc", self.accuracy(y_hat, batch[-1]), train=False)

    def accuracy(self, y_hat, y, averaged=True):
        """Compute the number of correct predictions.

        Defined in :numref:`sec_classification`"""
        y_hat = tf.reshape(y_hat, (-1, y_hat.shape[-1]))
        preds = tf.cast(tf.argmax(y_hat, axis=1), y.dtype)
        compare = tf.cast(preds == tf.reshape(y, -1), tf.float32)
        return tf.reduce_mean(compare) if averaged else compare

    def loss(self, y_hat, y, averaged=True):
        """Defined in :numref:`sec_softmax_concise`"""
        y_hat = tf.reshape(y_hat, (-1, y_hat.shape[-1]))
        y = tf.reshape(y, (-1,))
        fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        return fn(y, y_hat)

    def layer_summary(self, X_shape):
        """Defined in :numref:`sec_lenet`"""
        X = tf.random.normal(X_shape)
        for layer in self.net.layers:
            X = layer(X)
            print(layer.__class__.__name__, "output shape:\t", X.shape)
