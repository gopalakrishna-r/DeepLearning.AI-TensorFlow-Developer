from keras.layers import (
    BatchNormalization,
    SeparableConv2D,
    MaxPool2D,
    Activation,
    Flatten,
    Dropout,
    Dense,
)
import tensorflow as tf
import tf_slim as slim
from tensorflow import keras as K
from keras import Input
from tensorflow.python.framework import ops
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


@slim.add_arg_scope
def max_pool_2d(pool_size):
    return MaxPool2D(pool_size)


@slim.add_arg_scope
def dropout(dropout_rate):
    return Dropout(dropout_rate)


@slim.add_arg_scope
def separable_conv_blk(inputs, units, chan_dim, scope=None):
    inputs = SeparableConv2D(filters=units, kernel_size=(3, 3), padding="same")(inputs)
    inputs = Activation("relu")(inputs)
    inputs = BatchNormalization(axis=chan_dim)(inputs)
    return inputs


class CancerNet:
    @staticmethod
    def build(width, height, depth, classes):
        input_shape = (height, width, depth)
        chan_dim = -1

        if K.backend.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chan_dim = 1
        with slim.arg_scope([max_pool_2d], pool_size=(2, 2)):
            with slim.arg_scope([dropout], dropout_rate=0.25):
                # CONV => RELU => POOL
                inputs = Input(shape=input_shape, name="the_input_layer")
                outputs = separable_conv_blk(inputs, units=32, chan_dim=chan_dim)
                outputs = max_pool_2d()(outputs)
                outputs = dropout()(outputs)

                # (CONV => RELU => POOL) * 2
                outputs = slim.stack(
                    outputs,
                    separable_conv_blk,
                    [(64, chan_dim), (64, chan_dim)],
                    scope="separable_conv_blk_1",
                )
                outputs = max_pool_2d()(outputs)
                outputs = dropout()(outputs)

                # (CONV => RELU => POOL) * 3
                outputs = slim.stack(
                    outputs,
                    separable_conv_blk,
                    [(128, chan_dim), (128, chan_dim), (128, chan_dim)],
                    scope="separable_conv_blk_2",
                )
                outputs = max_pool_2d()(outputs)
                outputs = dropout()(outputs)

                # FC = > RELU
                outputs = Flatten()(outputs)
                outputs = Dense(256)(outputs)
                outputs = Activation("relu")(outputs)
                outputs = BatchNormalization()(outputs)
                outputs = dropout()(outputs)

                # classifier
                outputs = Dense(classes)(outputs)
                outputs = Activation("sigmoid")(outputs)

                return K.Model(inputs=inputs, outputs=outputs)
