import keras as keras
import numpy as np


class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (inputs - self.means_) / (self.stds_ + keras.backend.epsilon())
