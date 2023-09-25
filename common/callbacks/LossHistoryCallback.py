import keras
import tensorflow as tf
from matplotlib import pyplot as plt

from common.ProgressBoard import ProgressBoard
from common.Visualizer import Visualizer


def numpy(x, *args, **kwargs):
    return x.numpy(*args, **kwargs)


class LossHistory(keras.callbacks.Callback):
    def __init__(
        self,
        visualizer: Visualizer,
        max_epochs,
        num_train_batches,
        num_val_batches,
        plot_train_per_epoch=2,
        plot_valid_per_epoch=1,
    ):
        self.board = ProgressBoard()
        self.epoch = 0
        self.visualizer = visualizer
        self.max_epochs = max_epochs
        self.plot_train_per_epoch = plot_train_per_epoch
        self.plot_valid_per_epoch = plot_valid_per_epoch
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches
        self.board.xlim = [0, self.max_epochs]
        self.board.xlabel = "epoch"
        self.train_batch_idx = 0
        self.val_batch_idx = 0

    def on_epoch_end(self, epoch, logs=None):
        self.epoch += 1
        if self.epoch == self.max_epochs:
            self.plot()

    def on_train_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.build_plt_points("loss", logs.get("loss"), train=True)
        self.train_batch_idx += 1

    def on_test_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.build_plt_points("loss", logs.get("loss"), train=False)
        self.val_batch_idx += 1

    def build_plt_points(self, key, value, train):
        """calculate points for loss animation"""
        if train:
            x = self.train_batch_idx / self.num_train_batches
            n = self.num_train_batches / self.plot_train_per_epoch
        else:
            x = self.epoch + 1
            n = self.num_val_batches / self.plot_valid_per_epoch
        self.board.draw(x, value, ("train_" if train else "val_") + key, every_n=int(n))

    def plot(self):
        if not self.board.display:
            return
        if self.board.fig is None:
            self.board.fig = plt.figure(figsize=self.board.figsize)
        plt_lines, labels = [], []
        for (k, v), ls, color in zip(
            self.board.data.items(), self.board.ls, self.board.colors
        ):
            plt_lines.append(
                plt.plot([p.x for p in v], [p.y for p in v], linestyle=ls, color=color)[
                    0
                ]
            )
            labels.append(k)
        axes = self.board.axes or plt.gca()
        if self.board.xlim:
            axes.set_xlim(self.board.xlim)
        if self.board.ylim:
            axes.set_ylim(self.board.ylim)
        if not self.board.xlabel:
            self.board.xlabel = self.x
        axes.legend(plt_lines, labels)
        with self.visualizer.file_writer.as_default():
            tf.summary.image(
                "losses_vs_epoch", self.visualizer.plot_to_image(self.board.fig), step=0
            )
