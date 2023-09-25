import collections
import sys

from matplotlib_inline import backend_inline

from .HypeParams import HyperParameters

d2l = sys.modules[__name__]


def use_svg_display():
    """Use the svg format to display a plot in Jupyter.
    Defined in :numref:`sec_calculus`"""
    backend_inline.set_matplotlib_formats("svg")


def mean(x):
    return sum(x) / len(x)


class ProgressBoard(HyperParameters):
    """Plot data points in animation.

    Defined in :numref:`sec_oo-design`"""

    def __init__(
        self,
        xlabel=None,
        ylabel=None,
        xlim=None,
        ylim=None,
        xscale="linear",
        yscale="linear",
        ls=["-", "--", "-.", ":"],
        colors=["C0", "C1", "C2", "C3"],
        fig=None,
        axes=None,
        figsize=(3.5, 2.5),
        display=True,
    ):
        self.save_hyperparameters()

    def draw(self, x, y, label, every_n=1):
        raise NotImplementedError

    def draw(self, x, y, label, every_n=1):
        """Defined in :numref:`sec_utils`"""
        point = collections.namedtuple("point", ["x", "y"])
        if not hasattr(self, "raw_points"):
            self.raw_points = collections.OrderedDict()
            self.data = collections.OrderedDict()
        if label not in self.raw_points:
            self.raw_points[label] = []
            self.data[label] = []
        points = self.raw_points[label]
        line = self.data[label]
        points.append(point(x, y))
        if len(points) != every_n:
            return

        def mean(x):
            return sum(x) / len(x)

        line.append(point(mean([p.x for p in points]), mean([p.y for p in points])))
        points.clear()
