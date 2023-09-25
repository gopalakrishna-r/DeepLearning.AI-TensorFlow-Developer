import tensorflow as tf

from common.Module import Module


class LSTM(Module):
    def __init__(self, num_hiddens, normalizer):
        Module.__init__(self)
        self.save_hyperparameters()
        self.normalizer = normalizer
        self.rnn = tf.keras.layers.LSTM(
            num_hiddens, return_sequences=True, return_state=True, time_major=True
        )

    def forward(self, inputs, H_C=None):
        outputs, *H_C = self.rnn(self.normalizer(inputs), H_C)
        return outputs, H_C
