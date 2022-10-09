import keras.callbacks
import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbCallback

from utilites.preprocessor import batch_dataset, model_forecast
from utilites.visualizer import plot_series
from utilites.series_traits import trend, seasonality, noise, white_noise

if __name__ == '__main__':
    tf.keras.backend.clear_session()
    tf.random.set_seed(42)
    np.random.seed(33)
    with wandb.init(project="bidirectional_sweeps", name='lstm_sweeps',
                    save_code=True, entity='goofygc316', resume=True) as run:
        config = run.config
        time = np.arange(4 * 365 + 1, dtype='float32')
        baseline = 10
        amplitude = 40
        slope = 0.05
        noise_level = 5

        series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
        series += white_noise(time, noise_level)

        split_time = 1000
        time_train = time[:split_time]
        time_valid = time[split_time:]

        series_train = series[:split_time]
        series_valid = series[split_time:]

        window_size = 20
        shuffle_buffer_size = 1000

        x_series = batch_dataset(series_train, window_size=window_size,
                                 batch_size=16, shuffle_buffer_size=shuffle_buffer_size)

        model = tf.keras.models.Sequential([
            layers.Conv1D(filters=64, kernel_size=5,
                          strides=1, padding="causal",
                          activation='relu', input_shape=[None, 1]),
            layers.Bidirectional(layers.LSTM(32, return_sequences=True)),
            layers.Bidirectional(layers.LSTM(32)),
            layers.Dense(1),
            layers.Lambda(lambda x: x * 100.0)
        ])

        optimizer = tf.keras.optimizers.SGD(learning_rate=1e-5, momentum=0.9)
        model.compile(optimizer, metrics=['mae'], loss=tf.keras.losses.Huber())
        history = model.fit(x_series, epochs=500, callbacks=[WandbCallback()], workers=16, use_multiprocessing=True)

        forecasts = [
            model.predict(series[time:time + window_size][np.newaxis]) for time in range(len(series) - window_size)
        ]
        rnn_forecast = model_forecast(model, series[..., np.newaxis], window_size=window_size)
        rnn_forecast = rnn_forecast[split_time - window_size: -1, -1, 0]
        print(tf.keras.metrics.mean_absolute_error(series_valid, rnn_forecast).numpy())

        plt.plot(time_valid[0:], series_valid[0:], format, label=None)
        plt.plot(time_valid[0:], rnn_forecast[0:], format, label=None)
        plt.xlabel("Time")
        plt.ylabel("Series")
        plt.grid(True)
        run.log({"rnn_forecasts": plt})

        # plot mae and loss
        mae = history.history['mae']
        loss = history.history['loss']

        epochs = range(len(loss))
        plt.plot(epochs, mae, 'r')
        plt.plot(epochs, loss, 'b')
        plt.title('MAE and loss')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["MAE", "LOSS"])
        run.log({'MAE_LOSS': plt})

        # plot zoomed in MAE and loss
        epochs_zoom = epochs[200:]
        mae_zoom = mae[200:]
        loss_zoom = loss[200:]

        plt.plot(epochs_zoom, mae_zoom, 'r')
        plt.plot(epochs_zoom, loss_zoom, 'b')
        plt.title('MAE and loss zoomed in')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.legend(["MAE", "LOSS"])
        run.log({'MAE_LOSS_ZOOMED': plt})
