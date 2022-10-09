import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from wandb.integration.keras import WandbCallback

import wandb
from utilites.preprocessor import batch_dataset
from utilites.series_traits import trend, seasonality, white_noise
from utilites.visualizer import plot_series

if __name__ == '__main__':
    time = np.arange(4 * 365 + 1, dtype='float32')
    baseline = 10
    amplitude = 40
    slope = 0.05
    noise_level = 5

    series = baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
    noisy_series = white_noise(time, noise_level=5)
    series += noisy_series

    split_time = 1000
    time_train = time[:split_time]
    time_valid = time[split_time:]
    series_train = series[:split_time]
    series_valid = series[split_time:]
    with wandb.init(project="learning-rate-hyperparameter-sweeps", name='lr-rnn', entity='goofygc316') as run:
        config = run.config
        window_size = 20

        dataset = batch_dataset(series_train, batch_size=32, window_size=window_size, shuffle_buffer_size=1000)

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, input_shape=[window_size], activation='relu'),
            tf.keras.layers.Dense(10, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

        model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=8e-6, momentum=0.9),
                      loss='mse')
        history = model.fit(dataset, epochs=100,
                            callbacks=[WandbCallback()],
                            workers=6)

        forecasts = [
            model.predict(series[time:time + window_size][np.newaxis]) for time in
            range(len(series) - window_size)
        ]
        forecast = forecasts[split_time - window_size:]
        results = np.array(forecast)[:, 0, 0]
        plt.figure(figsize=[10, 6])
        plot_series(time_valid, series_valid)
        plot_series(time_valid, results)
        run.log({"mean_absolute_error": tf.keras.metrics.mean_absolute_error(series_valid, results).numpy()})
