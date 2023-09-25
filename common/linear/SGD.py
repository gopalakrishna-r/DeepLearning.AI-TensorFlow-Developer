from common.HypeParams import HyperParameters


class SGD(HyperParameters):  # @save
    """Minibatch stochastic gradient descent."""

    def __init__(self, lr):
        self.save_hyperparameters()

    def apply_gradients(self, grads_and_vars):
        for grad, param in grads_and_vars:
            param.assign_sub(self.lr * grad)
