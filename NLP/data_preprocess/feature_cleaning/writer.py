from contextlib import ExitStack

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow import feature_column as fc

Features = tf.train.Features
Feature = tf.train.Feature
FeatureList = tf.train.FeatureList
FeatureLists = tf.train.FeatureLists
Example = tf.train.Example
FloatList = tf.train.FloatList
BytesList = tf.train.BytesList
Int64List = tf.train.Int64List


def build_feature_column(x_train):
    age_mean, age_std = calculate_mean_stddev(x_train)
    housing_median_age, median_house_value = tf.feature_column.numeric_column(
        "housing_median_age", normalizer_fn=lambda x: (x - age_mean) / age_std
    ), fc.numeric_column("median_house_value")
    return [housing_median_age, median_house_value]


def save_to_multiple_files(name_prefix, data, n_parts=10):
    paths = [
        "{}.tfrecord-{:05d}-of-{:05d}".format(name_prefix, index, n_parts)
        for index in range(n_parts)
    ]
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path)) for path in paths]
        for index, (image, label) in data.enumerate():
            shard = index % n_parts
            example = create_example(image, label)
            writers[shard].write(example.SerializeToString())
    return paths


def create_example(image, label):
    image_data = tf.io.serialize_tensor(image)
    return Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),
                "label": Feature(int64_list=Int64List(value=[label])),
            }
        )
    )


def write_tf_records(x_train, y_train):
    with tf.io.TFRecordWriter("data_with_features.tfrecords") as f:
        for x, y in zip(x_train[:, 1:2], y_train):
            example = Example(
                features=Features(
                    feature={
                        "housing_median_age": Feature(float_list=FloatList(value=[x])),
                        "median_house_value": Feature(float_list=FloatList(value=[y])),
                    }
                )
            )
            f.write(example.SerializeToString())


def calculate_mean_stddev(x_train):
    scaler = StandardScaler()
    scaler.fit(x_train)
    x_mean = scaler.mean_
    x_stddev = scaler.scale_

    age_mean, age_std = x_mean[1], x_stddev[1]  # The median age is column in 1
    return age_mean, age_std


def mnist_dataset(
    filepaths,
    n_read_threads=5,
    shuffle_buffer_size=None,
    n_parse_threads=5,
    batch_size=32,
    cache=True,
):
    dataset = tf.data.TFRecordDataset(filepaths, num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)
    return dataset.prefetch(1)


def preprocess(
    tfrecord,
    feature_descriptions=dict(
        image=tf.io.FixedLenFeature([], tf.string, default_value=""),
        label=tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    ),
):
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    image = tf.reshape(image, shape=[28, 28])
    return image, example["label"]


def parse_examples(serial_examples, feature_descriptions):
    examples = tf.io.parse_example(serial_examples, feature_descriptions)
    targets = examples.pop("median_house_value")
    return examples, targets
