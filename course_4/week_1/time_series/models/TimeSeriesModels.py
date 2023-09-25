import tensorflow as tf
from common.Module import Module
from keras.callbacks import EarlyStopping


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class RNNForecaster(Module):
    def __init__(self, rnn):
        super().__init__()
        self.save_hyperparameters()

    def output_layer(self, inputs):
        return self.rnn(inputs)

    def forward(self, normalized_x, state=None):
        return self.output_layer(normalized_x)

    def configure_callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=3, mode="min"),
        ]

    def configure_loss(self):
        return tf.keras.losses.MeanSquaredError()

    def configure_optimizers(self):
        return tf.keras.optimizers.Adam()
