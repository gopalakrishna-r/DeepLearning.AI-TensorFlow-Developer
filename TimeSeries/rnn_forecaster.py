import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wandb.integration.keras import WandbCallback

import wandb
from utilites.preprocessor import batch_dataset
from utilites.series_traits import trend, seasonality, white_noise
from utilites.visualizer import plot_series

if __name__ == '__main__':
    time = np.arange(4 * 365 + 1)
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 15

    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    noisy_series = white_noise(time, noise_level=5)
    series += noisy_series

    split_time = 1000
    time_train = time[:split_time]
    time_valid = time[split_time:]
    series_train = series[:split_time]
    series_valid = series[split_time:]
    with wandb.init(project="learning-rate-hyperparameter-sweeps", name='lr-rnn',
                    save_code=True, entity='goofygc316', resume=True) as run:
        config = run.config

        window_size = 30
        dataset = batch_dataset(series_train, batch_size=32, window_size=window_size, shuffle_buffer_size=1000)
        model = tf.keras.models.Sequential([
            tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1), input_shape=[None]),
            tf.keras.layers.SimpleRNN(40, return_sequences=True),
            tf.keras.layers.SimpleRNN(40),
            tf.keras.layers.Dense(1),
            tf.keras.layers.Lambda(lambda x: x * 100.0)
        ])

        model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=5e-5, momentum=0.9),
            metrics = ['mae'],
            loss=tf.keras.losses.Huber())
        history = model.fit(dataset, epochs=500,
                            callbacks=[WandbCallback()],
                            workers=18)

        forecasts = [
            model.predict(series[time:time + window_size][np.newaxis]) for time in
            range(len(series) - window_size)
        ]
        forecast = forecasts[split_time - window_size:]
        results = np.array(forecast)[:, 0, 0]

        plt.figure(figsize=[10, 6])
        plt.plot(time_valid[0:], series_valid[0:], '-')
        plt.plot(time_valid[0:], results[0:], '-')
        plt.xlabel("Time")
        plt.ylabel("Series")
        plt.grid(True)
        run.log({'series_results': plt})

        # plot mae and loss
        mae = history.history['mae']
        loss = history.history['loss']

        epochs = range(len(loss))
        plt.plot(epochs, mae, 'r')
        plt.plot(epochs, loss, 'b')
        plt.title('MAE and loss')
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
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
        run.log({'MAE_LOSS_ZOOMED': plt})

