import tensorflow as tf

from common.Visualizer import numpy

from .HypeParams import HyperParameters
from .ProgressBoard import ProgressBoard


class Module(tf.keras.Model, HyperParameters):
    """The base class of models.

    Defined in :numref:`sec_oo-design`"""

    def __init__(self, plot_train_per_epoch=2, plot_valid_per_epoch=1):
        super().__init__()
        self.save_hyperparameters()
        self.board = ProgressBoard()
        self.training = None

    def loss(self, y_hat, y):
        raise NotImplementedError

    def build_model(self, train_dataset):
        raise NotImplementedError

    def configure_callbacks(self):
        raise NotImplementedError

    def forward(self, X):
        assert hasattr(self, "net"), "Neural network is defined"
        return self.net(X)

    def call(self, X, *args, **kwargs):
        if kwargs and "training" in kwargs:
            self.training = kwargs["training"]
        return self.forward(X, *args)

    def plot(self, key, value, train):
        """Plot a point in animation."""
        assert hasattr(self, "trainer"), "Trainer is not inited"
        self.board.xlabel = "epoch"
        if train:
            x = self.trainer.train_batch_idx / self.trainer.num_train_batches
            n = self.trainer.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.trainer.epoch + 1
            n = self.trainer.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(
            x, numpy(value), ("train_" if train else "val_") + key, every_n=int(n)
        )

    def configure_optimizers(self):
        raise NotImplementedError

    def save_model():
        raise NotImplementedError

    def load_model():
        raise NotImplementedError
