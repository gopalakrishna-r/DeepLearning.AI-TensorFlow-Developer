import tensorflow as tf

from .RNN import RNN
from ..Module import Module


class GRU(RNN):
    def __init__(self, num_inputs, num_hiddens):
        Module.__init__(self)
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.GRU(
            num_hiddens, return_sequences=True, return_state=True
        )

    def forward(self, X, state=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        rnn_outputs, *H_C = self.rnn(X, state)
        return self.output_layer(rnn_outputs)
