import argparse
import os

import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets

from NLP.data_preprocess.feature_cleaning.Standardization import Standardization
from NLP.data_preprocess.feature_cleaning.writer import (
    write_tf_records,
    parse_examples,
    build_feature_column,
    save_to_multiple_files,
    mnist_dataset,
)

Int64List = tf.train.Int64List
FloatList = tf.train.FloatList
BytesList = tf.train.BytesList

Features = tf.train.Features
Feature = tf.train.Feature
FeatureList = tf.train.FeatureList
FeatureLists = tf.train.FeatureLists
SequenceExample = tf.train.SequenceExample

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "feature_cleaning", help="comma separated list of models to use.", default="1"
    )
    args = parser.parse_args()
    if args.feature_cleaning:
        run_type = int(args.feature_cleaning)
        if run_type == 1:
            context = Features(
                feature={
                    "author_id": Feature(int64_list=Int64List(value=[123])),
                    "title": Feature(
                        bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])
                    ),
                    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25])),
                }
            )

            contents = [
                ["When", "shall", "we", "three", "meet", "again", "?"],
                ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"],
            ]
            comments = [
                ["When", "the", "hurlyburly", "'s", "done", "."],
                ["When", "the", "battle", "'s", "lost", "and", "won", "."],
            ]

            def words_to_feature(words):
                return Feature(
                    bytes_list=BytesList(value=[word.encode("utf-8") for word in words])
                )

            content_features = [words_to_feature(sentence) for sentence in contents]
            comment_features = [words_to_feature(sentence) for sentence in comments]

            sequence_example = SequenceExample(
                context=context,
                feature_lists=FeatureLists(
                    feature_list={
                        "content": FeatureList(feature=content_features),
                        "comments": FeatureList(feature=comment_features),
                    }
                ),
            )

            serialized_sequence_example = sequence_example.SerializeToString()

            context_feature_descriptions = {
                "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
                "title": tf.io.VarLenFeature(tf.string),
                "pub_date": tf.io.FixedLenFeature(
                    [3], tf.int64, default_value=[0, 0, 0]
                ),
            }
            sequence_feature_descriptions = {
                "content": tf.io.VarLenFeature(tf.string),
                "comments": tf.io.VarLenFeature(tf.string),
            }

            parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
                serialized_sequence_example,
                context_feature_descriptions,
                sequence_feature_descriptions,
            )

            print(parsed_context)
        elif run_type == 2:
            keras.backend.clear_session()
            np.random.seed(42)
            tf.random.set_seed(42)

            housing = fetch_california_housing()
            x_train_full, _, y_train_full, _ = train_test_split(
                housing.data, housing.target.reshape(-1, 1), random_state=42
            )
            x_train, _, y_train, _ = train_test_split(
                x_train_full, y_train_full, random_state=42
            )

            write_tf_records(x_train, y_train)

            batch_size = 32
            columns = build_feature_column(x_train)
            feature_descriptions = tf.feature_column.make_parse_example_spec(columns)
            dataset = tf.data.TFRecordDataset(["data_with_features.tfrecords"])
            dataset = (
                dataset.repeat()
                .shuffle(10000)
                .batch(batch_size)
                .map(lambda examples: parse_examples(examples, feature_descriptions))
            )

            columns_without_target = columns[:-1]
            model = keras.models.Sequential(
                [
                    keras.layers.DenseFeatures(feature_columns=columns_without_target),
                    keras.layers.Dense(1),
                ]
            )
            model.compile(
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"],
                loss="mse",
            )
            model.fit(
                dataset,
                steps_per_epoch=len(x_train) // batch_size,
                epochs=10,
                workers=16,
            )
        elif run_type == 3:
            (x_train_full, y_train_full), (
                x_test_full,
                y_test_full,
            ) = datasets.fashion_mnist.load_data()
            x_train, x_valid = x_train_full[:5000], x_train_full[5000:]
            y_train, y_valid = y_train_full[:5000], y_train_full[5000:]

            train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(
                len(x_train), name="train"
            )
            valid_set = tf.data.Dataset.from_tensor_slices(
                (x_valid, y_valid), name="valid"
            )
            test_set = tf.data.Dataset.from_tensor_slices(
                (x_test_full, y_test_full), name="test"
            )

            datasets = dict(train=train_set, valid=valid_set, test=test_set)

            dataset_filepaths = dict(
                map(
                    lambda item: (
                        item[0],
                        save_to_multiple_files(f"fashion_mnist.{item[0]}", item[1]),
                    ),
                    datasets.items(),
                )
            )

            tf_datasets = dict(
                map(
                    lambda item: (
                        item[0],
                        mnist_dataset(item[1])
                        if item[0] != "train"
                        else mnist_dataset(item[1], shuffle_buffer_size=60000),
                    ),
                    dataset_filepaths.items(),
                )
            )

            keras.backend.clear_session()
            tf.random.set_seed(42)
            np.random.seed(42)

            train_set = tf_datasets.get("train")

            standardization = Standardization(input_shape=[28, 28])

            sample_image_batches = train_set.take(100).map(lambda image, _: image)
            sample_images = np.concatenate(
                list(sample_image_batches.as_numpy_iterator()), axis=0
            ).astype(np.float32)
            standardization.adapt(sample_images)

            model = keras.models.Sequential(
                [
                    standardization,
                    keras.layers.Flatten(),
                    keras.layers.Dense(100, activation="relu"),
                    keras.layers.Dense(10, activation="softmax"),
                ]
            )

            model.compile(
                optimizer="nadam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )

            from datetime import datetime

            logs = os.path.join(
                os.curdir, "tf_logs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            tensorboard_cb = keras.callbacks.TensorBoard(
                log_dir=logs, histogram_freq=1, profile_batch=10
            )

            model.fit(
                train_set,
                epochs=5,
                validation_data=tf_datasets.get("valid"),
                workers=16,
                callbacks=[tensorboard_cb],
            )
