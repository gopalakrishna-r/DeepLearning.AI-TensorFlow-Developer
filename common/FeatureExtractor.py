# TensorFlow ≥2.0 is required
import tensorflow as tf
from common.Module import Module
import tensorflow_hub as hub


class TFFeatureExtractor(Module):
    def __init__(
        self,
        MODULE_HANDLE,
        fe_input_shape=None,
        fe_output_shape=None,
        perform_tuning=False,
        num_of_layers_if_trainable=None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.tf_feature_extractor = hub.KerasLayer(
            self.MODULE_HANDLE,
            dtype=tf.string,
            input_shape=self.fe_input_shape,
            output_shape=self.fe_output_shape,
            trainable=self.perform_tuning,
        )

        if perform_tuning:
            self.tf_feature_extractor.trainable = True
            if self.num_of_layers_if_trainable:
                for layer in self.tf_feature_extractor.layers[
                    -self.num_of_layers_if_trainable :
                ]:
                    layer.trainable = True
        else:
            self.tf_feature_extractor.trainable = False

    def call(self, X, *_):
        return self.tf_feature_extractor(X)
