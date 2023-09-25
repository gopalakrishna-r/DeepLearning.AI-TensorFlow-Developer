import tensorflow as tf

from common.Visualizer import Visualizer

from ..DataModule import DataModule


class ImageData(DataModule):
    """Defined in :numref:`sec_fashion_mnist`"""

    def __init__(self, visualizer: Visualizer):
        super().__init__()
        self.visualizer = visualizer
        self.save_hyperparameters()

    def text_labels(self, indices):
        raise NotImplementedError

    def labels(self):
        raise NotImplementedError

    def visualize(self, batch, nrows=1, ncols=8, labels=None):
        if labels is None:
            labels = []
        X, y = batch
        if not labels:
            labels = self.text_labels(y)
        self.visualizer.show_images(tf.squeeze(X), nrows, ncols, titles=labels)
