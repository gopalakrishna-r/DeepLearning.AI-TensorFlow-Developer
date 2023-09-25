import tensorflow as tf
from common.Trainer import Trainer


class TimeSeriesTrainer(Trainer):
    """Defined in :numref:`sec_oo-design`"""

    def __init__(self, visualizer, max_epochs=100, num_gpus=0):
        super().__init__(max_epochs=max_epochs, num_gpus=num_gpus)
        self.save_hyperparameters()
        self.visualizer = visualizer

        assert num_gpus == 0, "No GPU support yet"

    def prepare_model(self, rnn_forecaster, data):
        rnn_forecaster.trainer = self
        self.model = rnn_forecaster
        self.model.compile(
            loss=self.model.configure_loss(),
            optimizer=self.model.configure_optimizers(),
            metrics=[tf.keras.metrics.MeanAbsoluteError()],
        )

    def fit(self, forecaster, data):
        self.prepare_model(forecaster, data)
        self.history = self.model.fit(
            data.train,
            validation_data=data.valid,
            epochs=self.max_epochs,
            callbacks=forecaster.configure_callbacks(),
        )

    def evaluate(self, data):
        return self.model.evaluate(data)
