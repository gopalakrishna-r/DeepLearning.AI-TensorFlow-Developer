from .HypeParams import HyperParameters
from .ProgressBoard import ProgressBoard


class Trainer(HyperParameters):
    """The base class for training models with data.

    Defined in :numref:`subsec_oo-design-models`"""

    def __init__(self, max_epochs, num_gpus=0):
        self.save_hyperparameters()
        self.board = ProgressBoard()
        assert num_gpus == 0, "No GPU support yet"

    def prepare_data(self, data):
        raise NotImplementedError

    def prepare_dataset(self, *args):
        raise NotImplementedError

    def prepare_model(self, model):
        raise NotImplementedError

    def fit(self, model, data):
        raise NotImplementedError
