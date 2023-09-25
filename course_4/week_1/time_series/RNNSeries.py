from pathlib import Path
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from common.rnn.WindowGenerator import WindowGenerator

from .Preprocessor import ClimateDataProcessor


class RNNSeriesData(WindowGenerator):  # @save
    """The Text dataset."""

    def __init__(
        self, visualizer, input_width, label_width, shift, label_columns, batch_size
    ):
        self.save_hyperparameters()
        self.build()
        super().__init__(input_width, label_width, shift, label_columns)
        self.build_processors()

    def build(self):
        zip_path = tf.keras.utils.get_file(
            origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip",
            fname="jena_climate_2009_2016.csv.zip",
            extract=True,
            cache_subdir=Path.cwd(),
        )
        self.df = pd.read_csv(Path.cwd().joinpath(Path(zip_path).stem))
        self.slice_df = self.df.copy()[5::6]
        self.slice_df["Time_Stamp"] = pd.to_datetime(
            self.slice_df.pop("Date Time"), format="%d.%m.%Y %H:%M:%S"
        ).map(pd.Timestamp.timestamp)
        self.ds_size = len(self.slice_df)
        self.train_df = self.slice_df[: int(self.ds_size * 0.7)]
        self.valid_df = self.slice_df[int(self.ds_size * 0.7) : int(self.ds_size * 0.9)]
        self.test_df = self.slice_df[int(self.ds_size * 0.9) :]

    def build_processors(self):
        self.preprocessor = ClimateDataProcessor(column_indices=self.column_indices)
        self.normalizer = tf.keras.layers.Normalization(axis=-1)
        self.normalizer.adapt(
            tf.data.Dataset.from_tensor_slices(np.array(self.train_df))
            .window(size=1, shift=1)
            .flat_map(lambda window: window.batch(1))
            .batch(64, drop_remainder=True)
            .map(self.preprocessor)
        )

    def make_dataset(self, data, batch_size, total_window_size, shuffle=False):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=total_window_size,
            sequence_stride=1,
            shuffle=shuffle,
            batch_size=batch_size,
        )
        return (
            ds.map(self.preprocessor, num_parallel_calls=tf.data.AUTOTUNE)
            .map(self.normalizer, num_parallel_calls=tf.data.AUTOTUNE)
            .map(self.split_window, num_parallel_calls=tf.data.AUTOTUNE)
        )

    @property
    def train(self):
        return self.make_dataset(
            self.train_df,
            batch_size=self.batch_size,
            total_window_size=self.total_window_size,
            shuffle=True,
        )

    @property
    def valid(self):
        return self.make_dataset(
            self.valid_df,
            batch_size=self.batch_size,
            total_window_size=self.total_window_size,
        )

    @property
    def test(self):
        return self.make_dataset(
            self.test_df,
            batch_size=self.batch_size,
            total_window_size=self.total_window_size,
        )

    def visualize_df(self, plot_cols):
        plot_features = self.ds.numpy()[plot_cols]
        plot_features.index = self.index
        _ = plot_features.plot(subplots=True)

        plot_features = self.df[plot_cols][:480]
        plot_features.index = self.index[:480]
        _ = plot_features.plot(subplots=True)

    def plot(
        self,
        model=None,
        plot_col="T (degC)",
        max_subplots=3,
        plot_title="image_grid_preds",
    ):
        inputs, labels = self.example
        figure = plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f"{plot_col} [normed]")
            plt.plot(
                self.input_indices,
                inputs[n, :, plot_col_index],
                label="Inputs",
                marker=".",
                zorder=-10,
            )

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(
                self.label_indices,
                labels[n, :, label_col_index],
                edgecolors="k",
                label="Labels",
                c="#2ca02c",
                s=64,
            )
            if model is not None:
                predictions = model(inputs)
                plt.scatter(
                    self.label_indices,
                    predictions[n, :, label_col_index],
                    marker="X",
                    edgecolors="k",
                    label="Predictions",
                    c="#ff7f0e",
                    s=64,
                )

            if n == 0:
                plt.legend()

        plt.xlabel("Time [h]")
        with self.visualizer.file_writer.as_default():
            tf.summary.image(plot_title, self.visualizer.plot_to_image(figure), step=0)
