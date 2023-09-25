import contextlib
import io
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def numpy(x, *args, **kwargs):
    return x.numpy(*args, **kwargs)


class Visualizer:
    def __init__(self):
        self.plot_dir = self.clean_create_save_dir()
        self.file_writer = tf.summary.create_file_writer(str(self.plot_dir.resolve()))

    def clean_create_save_dir(self):
        def rm_tree(pth):
            pth = Path(pth)
            for child in pth.iterdir():
                if child.is_file():
                    child.unlink()
                else:
                    rm_tree(child)
            pth.rmdir()

        if not (logdir := Path.joinpath(Path.cwd(), Path("logs"))).exists():
            # rm_tree(str(logdir.resolve()))
            Path.mkdir(logdir, parents=True, exist_ok=True)
        self.logdir = logdir

        # Clear out prior logging data.
        if (plot_dir := Path.joinpath(logdir, "plots")).exists():
            rm_tree(str(plot_dir.resolve()))

        plot_dir = Path.joinpath(
            plot_dir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        return plot_dir

    def plot_img_grid(self, imgs, num_rows, num_cols, titles=None, scale=6):
        """Plot a list of images.

        Defined in :numref:`sec_utils`"""
        figure = plt.figure()
        figsize = (num_cols * scale, num_rows * scale)
        _, axes = plt.subplots(num_rows, num_cols, squeeze=False, figsize=figsize)
        axes = axes.flatten()
        for i, (ax, img) in enumerate(zip(axes, imgs)):
            with contextlib.suppress(Exception):
                img = numpy(img)
            ax.imshow(img)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            if titles:
                ax.set_title(titles[i])
        with self.file_writer.as_default():
            tf.summary.image("image_grid", self.plot_to_image(figure), step=0)

    def show_images(self, images, cols=1, titles=None, image_title="examples.png"):
        assert titles is None or len(images) == len(titles)
        n_images = len(images)
        if titles is None:
            titles = ["Image (%d)" % i for i in range(1, n_images + 1)]
        fig = plt.figure()
        for n, (image, title) in enumerate(zip(images, titles)):
            a = fig.add_subplot(cols, int(np.ceil(n_images / float(cols))), n + 1)
            if image.ndim == 2:
                plt.gray()
            a.set_title(title, fontsize=15 * int(np.ceil(n_images / float(cols))))
        fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
        with self.file_writer.as_default():
            tf.summary.image(image_title, self.plot_to_image(fig), step=0)

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def set_figsize(self, figsize=(3.5, 2.5)):
        """Set the figure size for matplotlib.

        Defined in :numref:`sec_calculus`"""
        plt.rcParams["figure.figsize"] = figsize

    def set_axes(self, axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
        """Set the axes for matplotlib.

        Defined in :numref:`sec_calculus`"""
        axes.set_xlabel(xlabel), axes.set_ylabel(ylabel)
        axes.set_xscale(xscale), axes.set_yscale(yscale)
        axes.set_xlim(xlim), axes.set_ylim(ylim)
        if legend:
            axes.legend(legend)
        axes.grid()

    def plot(
        self,
        X,
        Y=None,
        xlabel=None,
        ylabel=None,
        legend=[],
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        fmts=("-", "m--", "g-.", "r:"),
        figsize=(3.5, 2.5),
        axes=None,
        plot_name="",
    ):
        """Plot data points.

        Defined in :numref:`sec_calculus`"""

        def has_one_axis(X):  # True if X (tensor or list) has 1 axis
            return (
                hasattr(X, "ndim")
                and X.ndim == 1
                or isinstance(X, list)
                and not hasattr(X[0], "__len__")
            )

        figure = plt.figure()
        if has_one_axis(X):
            X = [X]
        if Y is None:
            X, Y = [[]] * len(X), X
        elif has_one_axis(Y):
            Y = [Y]
        if len(X) != len(Y):
            X = X * len(Y)

        self.set_figsize(figsize)
        if axes is None:
            axes = plt.gca()
        axes.cla()
        for x, y, fmt in zip(X, Y, fmts):
            axes.plot(x, y, fmt) if len(x) else axes.plot(y, fmt)
        self.set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        with self.file_writer.as_default():
            tf.summary.image(plot_name, self.plot_to_image(figure), step=0)
