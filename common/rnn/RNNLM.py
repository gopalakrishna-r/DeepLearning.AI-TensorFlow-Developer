import tensorflow as tf

from .RNNLMScratch import RNNLMScratch
from ..utilities import transpose


class RNNLM(RNNLMScratch):
    """The RNN-based language model implemented with high-level APIs.

    Defined in :numref:`sec_rnn-concise`"""

    def init_params(self):
        self.linear = tf.keras.layers.Dense(self.vocab_size)

    def output_layer(self, hiddens):
        print(f"Module RNNLM output_layer {hiddens}")
        return transpose(self.linear(hiddens), (1, 0, 2))
