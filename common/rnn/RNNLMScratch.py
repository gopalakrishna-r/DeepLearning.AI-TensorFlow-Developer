import tensorflow as tf

from ..linear.Classifier import Classifier
from ..callbacks.LossHistoryCallback import LossHistory
from ..utilities import tensor, reshape, argmax


class RNNLMScratch(Classifier):  # @save
    """The RNN-based language model implemented from scratch."""

    def __init__(self, rnn, visualizer, vocab_size, lr=0.01, clipvalue=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.init_params()
        self.visualizer = visualizer

    def init_params(self):
        self.W_hq = tf.Variable(
            tf.random.normal((self.rnn.num_hiddens, self.vocab_size)) * self.rnn.sigma
        )
        self.b_q = tf.Variable(tf.zeros(self.vocab_size))

    def one_hot(self, X):
        # Output shape: (num_steps, batch_size, vocab_size)
        return tf.one_hot(tf.transpose(X), self.vocab_size)

    def output_layer(self, rnn_outputs):
        outputs = [tf.matmul(H, self.W_hq) + self.b_q for H in rnn_outputs]
        return tf.stack(outputs, 1)

    def forward(self, X, state=None):
        embs = self.one_hot(X)
        rnn_outputs, _ = self.rnn(embs, state)
        return self.output_layer(rnn_outputs)

    def predict(self, prefix, num_preds, vocab, device=None):
        """Defined in :numref:`sec_rnn-scratch`"""
        state, outputs = None, [vocab[prefix[0]]]
        for i in range(len(prefix) + num_preds - 1):
            X = tensor([[outputs[-1]]])
            embs = self.one_hot(X)
            rnn_outputs, state = self.rnn(embs, state)
            if i < len(prefix) - 1:  # Warm-up period
                outputs.append(vocab[prefix[i + 1]])
            else:  # Predict num_preds steps
                Y = self.output_layer(rnn_outputs)
                outputs.append(int(reshape(argmax(Y, axis=2), 1)))
        return "".join([vocab.idx_to_token[i + 1] for i in outputs])

    def configure_callbacks(self, max_epochs, num_train_batches, num_val_batches):
        return LossHistory(
            self.visualizer, max_epochs, num_train_batches, num_val_batches
        )

    def configure_loss(self):
        return tf.keras.losses.SparseCategoricalCrossentropy()
