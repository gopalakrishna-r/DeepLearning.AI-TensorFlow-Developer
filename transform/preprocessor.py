import itertools
import math
from pathlib import Path

import fuzzywuzzy.process
import keras.layers as layers
import pandas as pd
import tensorflow as tf
from keras.layers import StringLookup, IntegerLookup
from tensorflow import float32, string

EMBEDDING_DIMS = 29
DISCRETE_DIMS = 891


def load_playground_data():
    data = pd.read_csv(str(Path.cwd().joinpath("dataset/data.csv").resolve()))
    submission_data = pd.read_csv(
        str(Path.cwd().joinpath("dataset/sample_submission.csv").resolve()),
        index_col="row-col",
    )
    return data, submission_data


def get_missings(df):
    labels, values = zip(
        *[
            (column, (column_null_sum / len(df[column])) * 100)
            for column in df.columns
            if (column_null_sum := df[column].isnull().sum())
        ]
    )
    missings = pd.DataFrame({"Column": labels, "Missing(Percent)": values}).sort_values(
        by="Missing(Percent)", ascending=False
    )
    return missings


def high_correlated(col, data):
    return (
        data.corrwith(data[col])
        .abs()
        .sort_values(ascending=False)[1:30]
        .index.to_list()
    )


def rmse(y_true, y_pred):
    return tf.keras.backend.sqrt(
        tf.keras.backend.mean(tf.keras.backend.square(y_pred - y_true))
    )


def replace_matches_in_column(df, column, string_to_match, min_ratio=47):
    strings = df[column].unique()
    matches = fuzzywuzzy.process.extract(
        string_to_match, strings, limit=10, scorer=fuzzywuzzy.fuzz.token_sort_ratio
    )

    close_matches = [matches[0] for match in matches if match[1] >= min_ratio]

    rows_with_matches = df[column].isin(close_matches)
    df.loc[rows_with_matches, column] = string_to_match


def df_to_dataset(dataframe):
    df = dataframe.copy()
    labels = df.pop("target")
    ds = tf.data.Dataset.from_tensor_slices(dict(df), labels)
    ds = ds.shuffle(
        buffer_size=len(df),
    )
    ds = ds.batch(32)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


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
        column_names=csv_headers,
        column_defaults=col_defaults,
        label_name=label_names,
        num_epochs=1,
        header=True,
        shuffle=shuffle,
        prefetch_buffer_size=tf.data.AUTOTUNE,
        num_parallel_reads=tf.data.AUTOTUNE,
    )
    return dataset


def morph_csv_to_dataset(csv_data, batch_size):
    return (
        tf.data.Dataset.from_tensor_slices(csv_data)
        .batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        .shuffle(buffer_size=100)
        .prefetch(tf.data.experimental.AUTOTUNE)
    )


def create_model_inputs(
    categorical_features_names, textual_features_names, numerical_feature_names
):
    inputs = {
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(), name=feature_name, dtype=string),
                ),
                categorical_features_names,
            )
        ),
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(), name=feature_name, dtype=float32),
                ),
                numerical_feature_names,
            )
        ),
        **dict(
            map(
                lambda feature_name: (
                    feature_name,
                    tf.keras.Input(shape=(), name=feature_name, dtype=string),
                ),
                textual_features_names,
            )
        ),
    }
    return inputs


def encode_inputs(
    inputs,
    categorical_features,
    categorical_features_with_vocab,
    use_embedding=False,
):
    encoded_features = []
    for feature_name in inputs:
        if feature_name in categorical_features:
            vocab = categorical_features_with_vocab[feature_name]
            lookup = StringLookup(
                vocabulary=vocab,
                mask_token=None,
                num_oov_indices=0,
                output_mode="int" if use_embedding else "binary",
            )
            if use_embedding:
                encoded_feature = lookup(inputs[feature_name])
                embedding_dims = int(math.sqrt((len(vocab))))
                embedding = layers.Embedding(
                    input_dim=len(vocab), output_dim=embedding_dims
                )
                encoded_feature = embedding(encoded_feature)
            else:
                encoded_feature = lookup(tf.expand_dims(inputs[feature_name], -1))
        else:
            encoded_feature = tf.expand_dims(inputs[feature_name], -1)

        encoded_features.append(encoded_feature)
    return layers.concatenate(encoded_features)


def encode_inputs_with_lookups(inputs, csv_feature_names):
    numerical_feature_vocab_cont = csv_feature_names[
        "NUMERIC_FEATURES_WITH_VOCABULARY"
    ]["continous"]
    numerical_feature_vocab_non_cont = csv_feature_names[
        "NUMERIC_FEATURES_WITH_VOCABULARY"
    ]["noncontinous"]
    categorical_feature_vocab_ = csv_feature_names[
        "CATEGORICAL_FEATURES_WITH_VOCABULARY"
    ]
    text_categorical_feature_vocab_ = csv_feature_names[
        "TEXT_FEATURE_NAMES_WITH_VOCABULARY"
    ]
    encoded_features = list(
        itertools.chain.from_iterable(
            [
                build_feature_with_vocab(
                    categorical_feature_vocab_, build_text_feature, inputs
                ),
                build_feature_with_vocab(
                    text_categorical_feature_vocab_, build_text_feature, inputs
                ),
                build_feature_with_vocab(
                    numerical_feature_vocab_cont, build_discretized_features, inputs
                ),
                build_feature_with_vocab(
                    numerical_feature_vocab_non_cont,
                    build_numerical_feature_non_cont,
                    inputs,
                ),
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


def build_text_feature(feature_name, inputs, vocab):
    lookup = StringLookup(
        vocabulary=vocab, mask_token=None, num_oov_indices=1, output_mode="int"
    )
    encoded_feature = lookup(inputs[feature_name])
    embedding_dims = int(math.sqrt((len(vocab))))
    embedding = layers.Embedding(input_dim=len(vocab), output_dim=embedding_dims)
    encoded_feature = embedding(encoded_feature)
    return encoded_feature


def build_discretized_features(feature_name, inputs, data):
    discretizer = layers.Discretization(num_bins=DISCRETE_DIMS)
    discretizer.adapt(data)
    encoded_feature = discretizer(inputs[feature_name])
    embedding = layers.Embedding(input_dim=DISCRETE_DIMS, output_dim=EMBEDDING_DIMS)
    encoded_feature = embedding(encoded_feature)
    return encoded_feature


def build_numerical_feature_non_cont(feature_name, inputs, vocab):
    lookup = IntegerLookup(
        vocabulary=vocab,
        num_oov_indices=1,
        output_mode="int",
    )
    encoded_feature = lookup(inputs[feature_name])
    embedding_dims = int(math.sqrt((len(vocab))))
    embedding = layers.Embedding(input_dim=len(vocab), output_dim=embedding_dims)
    encoded_feature = embedding(encoded_feature)
    return encoded_feature
