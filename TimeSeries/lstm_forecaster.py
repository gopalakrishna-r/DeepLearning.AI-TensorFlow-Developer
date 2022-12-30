import keras.layers as layers
import keras.layers as layers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import wandb

from TimeSeries.utilites.preprocessor import batch_dataset
from utilites.series_traits import trend, seasonality, white_noise

if __name__ == "__main__":
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(33)
    with wandb.init(
        project="bidirectional_sweeps",
        name="lstm_sweeps",
        save_code=True,
        entity="goofygc316",
        resume=True,
    ) as run:
        time = np.arange(4 * 365 + 1, dtype="float32")
        baseline = 10
        amplitude = 40
        slope = 0.05
        noise_level = 5

        series = (
            baseline
            + trend(time, slope)
            + seasonality(time, period=365, amplitude=amplitude)
        )
        series += white_noise(time, noise_level)

        split_time = 1000
        time_train = time[:split_time]
        time_valid = time[split_time:]

        series_train = series[:split_time]
        series_valid = series[split_time:]

        window_size = 20
        batch_size = 32
        shuffle_buffer_size = 1000

        x_series = batch_dataset(
            series_train,
            window_size=window_size,
            batch_size=batch_size,
            shuffle_buffer_size=shuffle_buffer_size,
        )

        model = tf.keras.models.Sequential(
            [
                layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
                layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
                layers.Bidirectional(layers.LSTM(32)),
                layers.Dense(1),
                layers.Lambda(lambda x: x * 100.0),
            ]
        )

        optimizer = tf.keras.optimizers.SGD(lr=1e-5, momentum=0.9)
        model.compile(optimizer, metrics=["mae"], loss=tf.keras.losses.Huber())
        history = model.fit(
            x_series, epochs=100, callbacks=[], workers=16, use_multiprocessing=True
        )

        forecasts = [
            model.predict(series[time : time + window_size][np.newaxis])
            for time in range(len(series) - window_size)
        ]
        forecast = forecasts[split_time - window_size :]
        results = np.array(forecast)[:, 0, 0]

        print(tf.keras.metrics.mean_absolute_error(series_valid, results).numpy())

        # plot mae and loss
        mae = history.history["mae"]
        loss = history.history["loss"]

        epochs = range(len(loss))
        plt.plot(epochs, mae, "r")
        plt.plot(epochs, loss, "b")
        plt.title("MAE and loss")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        run.log({"MAE_LOSS": plt})

        # plot zoomed in MAE and loss
        epochs_zoom = epochs[200:]
        mae_zoom = mae[200:]
        loss_zoom = loss[200:]

        plt.plot(epochs_zoom, mae_zoom, "r")
        plt.plot(epochs_zoom, loss_zoom, "b")
        plt.title("MAE and loss zoomed in")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        run.log({"MAE_LOSS_ZOOMED": plt})
