from pathlib import Path

import keras
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import optimizers, losses

from credit_csv_keys import (
    NUMERICAL_CAT_CONT_KEYS,
    TARGET_FEATURE_NAME,
)
from experiment_utils import (
    create_bl_model,
    build_column_defaults,
)
from preprocessor import fetch_dataset_from_csv

CSV_PATH = str(Path.cwd().parent.joinpath("transform/dataset/credit/creditcard.csv").resolve())

CSV_HEADER = NUMERICAL_CAT_CONT_KEYS + [TARGET_FEATURE_NAME]

COLUMN_DEFAULTS = build_column_defaults(
    NUMERICAL_CAT_CONT_KEYS,
    [],
    TARGET_FEATURE_NAME,
    csv_headers=CSV_HEADER,
)

batch_size = 128

csv_dataset = fetch_dataset_from_csv(
    csv_filepath=CSV_PATH,
    batch_size=batch_size,
    csv_headers=CSV_HEADER,
    col_defaults=COLUMN_DEFAULTS,
    label_names=TARGET_FEATURE_NAME,
    shuffle=True,
)


def is_test(x, y):
    return x % 5 == 0


def is_train(x, y):
    return not is_test(x, y)


recover = lambda _, y: y

val_dataset = csv_dataset.enumerate() \
    .filter(is_test) \
    .map(recover)

train_dataset = csv_dataset.enumerate() \
    .filter(is_train) \
    .map(recover)

num_classes = 2


@tf.function
def count_class(count, batch):
    y, _, c = tf.unique_with_counts(batch[1])
    return tf.tensor_scatter_nd_add(count, tf.expand_dims(y, axis=1), c)


counts = train_dataset.reduce(
    initial_state=tf.zeros(num_classes, tf.int32),
    reduce_func=count_class)

# Compute the imbalance ratio for each class
total_instances = tf.reduce_sum(counts)
class_0_ratio = counts[0] / total_instances
class_1_ratio = counts[1] / total_instances

print("Class 0 ratio:", class_0_ratio.numpy())
print("Class 1 ratio:", class_1_ratio.numpy())

weight_for_0 = 1.0 / counts[0].numpy()
weight_for_1 = 1.0 / counts[1].numpy()

train_dataset_unbatched = train_dataset.unbatch()
TEXT_FEATURES_WITH_VOCABULARY = {}

NUMERIC_FEATURES_WITH_VOCAB = dict(
    continous=dict(
        map(
            lambda feature: (
            feature, train_dataset_unbatched.map(lambda x, _: x[feature]).map(lambda x: tf.expand_dims(x, -1))),
            NUMERICAL_CAT_CONT_KEYS,
        )
    ),
    noncontinous={},
)

CATEGORICAL_FEATURES_WITH_VOCABULARY = {}

CSV_FEATURES = {
    "NUMERIC_FEATURES_WITH_VOCABULARY": NUMERIC_FEATURES_WITH_VOCAB,
    "CATEGORICAL_FEATURES_WITH_VOCABULARY": CATEGORICAL_FEATURES_WITH_VOCABULARY,
    "TEXT_FEATURE_NAMES_WITH_VOCABULARY": TEXT_FEATURES_WITH_VOCABULARY,
    "TARGET_FEATURE_NAME": TARGET_FEATURE_NAME,
}

dropout_rate = 0.1
hidden_units = [256, 256]
learning_rate = 1e-2

baseline_model = create_bl_model(
    hidden_units=hidden_units,
    dropout=dropout_rate,
    csv_feature_names=CSV_FEATURES
)

metrics = [
    keras.metrics.FalseNegatives(name="fn"),
    keras.metrics.FalsePositives(name="fp"),
    keras.metrics.TrueNegatives(name="tn"),
    keras.metrics.TruePositives(name="tp"),
    keras.metrics.Precision(name="precision"),
    keras.metrics.Recall(name="recall"),
]
baseline_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=metrics,
)

callbacks = [keras.callbacks.ModelCheckpoint("fraud_model_at_epoch_{epoch}.h5")]
class_weight = {0: weight_for_0, 1: weight_for_1}

history = baseline_model.fit(
    train_dataset, epochs=50, verbose=2, workers=-1,
    use_multiprocessing=True, validation_data=val_dataset,
    class_weight=class_weight
)

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
