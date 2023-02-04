import argparse
import os

import keras.losses
import matplotlib.pyplot as plt
import tensorflow as tf
from imutils import paths
from keras.callbacks import EarlyStopping
from keras.optimizers import Adagrad
from keras.utils import to_categorical

from CancerNet import CancerNet
import config


def load_images(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.resize(image, config.IMAGE_SIZE)

    label = tf.strings.split(image_path, os.path.sep)[-2]
    label = tf.strings.to_number(label, tf.int32)

    return image, label


@tf.function
def augment(image, _):
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_flip_left_right(image)

    return image, _


ap = argparse.ArgumentParser()
ap.add_argument("-p", "--plot", type=str, default='plot.png', help='path to output loss/accuracy plt')
args = vars(ap.parse_args())

train_paths = list(paths.list_images(config.TRAIN_PATH))
val_paths = list(paths.list_images(config.VAL_PATH))
test_paths = list(paths.list_images(config.TEST_PATH))

train_labels = [int(p.split(os.path.sep)[-2]) for p in train_paths]
train_labels = to_categorical(train_labels)
class_totals = train_labels.sum(axis=0)
class_weights = {i: class_totals.max() / class_totals[i] for i in range(0, len(class_totals))}

train_dataset = tf.data.Dataset.from_tensor_slices(train_paths)
train_dataset = (train_dataset.
                 shuffle(len(train_paths)).
                 map(load_images, num_parallel_calls=tf.data.AUTOTUNE).
                 map(augment, num_parallel_calls=tf.data.AUTOTUNE).
                 cache().
                 batch(batch_size=config.BS, num_parallel_calls=tf.data.AUTOTUNE).
                 prefetch(tf.data.AUTOTUNE)
                 )

val_dataset = tf.data.Dataset.from_tensor_slices(val_paths)
val_dataset = (val_dataset.
               map(load_images, num_parallel_calls=tf.data.AUTOTUNE).
               cache().
               batch(batch_size=config.BS, num_parallel_calls=tf.data.AUTOTUNE).
               prefetch(tf.data.AUTOTUNE)
               )

test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = (test_dataset.
                map(load_images, num_parallel_calls=tf.data.AUTOTUNE).
                cache().
                batch(batch_size=config.BS, num_parallel_calls=tf.data.AUTOTUNE).
                prefetch(tf.data.AUTOTUNE)
                )

model = CancerNet.build(width=48, height=48, depth=3, classes=1)
optimizer = Adagrad(lr=config.INIT_LR, decay=config.INIT_LR / config.NUM_EPOCHS)
model.compile(loss=keras.losses.BinaryCrossentropy(), optimizer=optimizer, metrics=['accuracy'])

es = EarlyStopping(
    monitor="val_loss",
    patience=config.EARLY_STOPPING_PATIENCE,
    restore_best_weights=True)

history = model.fit(
    x=train_dataset,
    validation_data=val_dataset,
    class_weight=class_weights,
    epochs=config.NUM_EPOCHS,
    callbacks = [es],
    verbose=2)

(_, acc) = model.evaluate(test_dataset)
print(f'accuracy of test set :{acc * 100}.2f%')

plt.style.use("ggplot")
plt.figure()
plt.plot(history.history["loss"], label="train_loss")
plt.plot(history.history["val_loss"], label="val_loss")
plt.plot(history.history["accuracy"], label="train_acc")
plt.plot(history.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
