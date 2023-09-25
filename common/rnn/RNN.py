import tensorflow as tf

from ..Module import Module


class RNN(Module):  # @save
    """The RNN model implemented with high-level APIs."""

    def __init__(self, num_hiddens):
        super().__init__()
        self.save_hyperparameters()
        self.rnn = tf.keras.layers.SimpleRNN(
            num_hiddens, return_sequences=True, return_state=True, time_major=True
        )

    def forward(self, inputs, H=None):
        print(f"Module RNN forward {inputs}")
        outputs, H = self.rnn(inputs, H)
        return outputs, H
