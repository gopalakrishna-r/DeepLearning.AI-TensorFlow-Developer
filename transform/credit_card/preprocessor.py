import itertools
import math
from pathlib import Path

import fuzzywuzzy.process
import keras.layers as layers
import pandas as pd
import tensorflow as tf
from keras.layers import StringLookup, IntegerLookup, Normalization
from tensorflow import float32, string

EMBEDDING_DIMS = 29
DISCRETE_DIMS = 891


def fetch_dataset_from_csv(
        csv_filepath,
        batch_size,
        csv_headers=None,
        col_defaults=None,
        label_names=None,
        shuffle=False,
):
    dataset = tf.data.experimental.make_csv_dataset(
        csv_filepath,
        batch_size=batch_size,
        select_columns=csv_headers,
        column_defaults=col_defaults,
        label_name=label_names,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    return dataset


def create_model_inputs(
        categorical_features_names, textual_features_names, numerical_feature_names
):
    inputs = {
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(1,), name=feature_name, dtype=string),
                ),
                categorical_features_names,
            )
        ),
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(1,), name=feature_name, dtype=float32),
                ),
                numerical_feature_names,
            )
        ),
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(1,), name=feature_name, dtype=string),
                ),
                textual_features_names,
            )
        ),
    }
    return inputs


def encode_inputs_with_lookups(inputs, csv_feature_names):
    numerical_feature_vocab_cont = csv_feature_names[
        "NUMERIC_FEATURES_WITH_VOCABULARY"
    ]["continous"]
    encoded_features = list(
        itertools.chain.from_iterable(
            [
                build_feature_with_vocab(
                    numerical_feature_vocab_cont, build_numerical_feature, inputs
                )
            ]
        )
    )
    return layers.concatenate(encoded_features)


def build_feature_with_vocab(feature_vocabs, feature_morpher, inputs):
    return list(
        map(
            lambda feature_name: feature_morpher(
                feature_name,
                inputs,
                feature_vocabs[feature_name],
            ),
            feature_vocabs.keys(),
        )
    )


def build_numerical_feature(feature_name, inputs, vocab):
    # Create a Normalization layer for our feature
    normalizer = Normalization()
    # Learn the statistics of the data
    normalizer.adapt(vocab)
    # Normalize the input feature
    encoded_feature = normalizer(inputs[feature_name])
    return encoded_feature
