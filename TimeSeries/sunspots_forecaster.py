from pathlib import Path

import numpy as np
import tensorflow.keras as keras
import wandb
from wandb.integration.keras import WandbCallback

from utilites.dataloader import process_csv_data
from utilites.preprocessor import batch_csv_dataset, model_forecast

if __name__ == "__main__":
    MODEL_NAME = "sunspots_forecaster"
    SAVE_MODEL_DIR = Path.cwd().joinpath("models")
    csv_data, header = process_csv_data()
    sunspots, time_steps = zip(*[(float(row[2]), int(row[0])) for row in csv_data])
    series, time = np.array(sunspots), np.array(time_steps)

    split_time = 3000
    time_train = time[:split_time]
    x_train = series[:split_time]
    time_valid = time[split_time:]
    x_valid = series[split_time:]

    with wandb.init(
        project="sunspots_forecaster", name="lr-rnn", entity="goofygc316", resume=True
    ) as run:
        config = run.config
        run.define_metric("mean_absolute_error", summary="min")

        train_dataset = batch_csv_dataset(
            x_train,
            window_size=config.window_size,
            batch_size=config.batch_size,
            shuffle_buffer_size=1000,
            cache=True,
        )
        valid_dataset = batch_csv_dataset(
            x_valid,
            window_size=config.window_size,
            batch_size=config.batch_size,
            shuffle_buffer_size=1000,
            cache=True,
        )

        model = keras.models.Sequential(
            [
                keras.layers.Conv1D(
                    filters=32,
                    kernel_size=5,
                    strides=1,
                    padding="causal",
                    activation="relu",
                    input_shape=[None, 1],
                ),
                keras.layers.LSTM(config.filters_1, return_sequences=True),
                keras.layers.LSTM(config.filters_1, return_sequences=True),
                keras.layers.Dense(config.filters_2, activation="relu"),
                keras.layers.Dense(config.filters_2, activation="relu"),
                keras.layers.Dense(1, activation="relu"),
                keras.layers.Lambda(lambda x: x * 400),
            ]
        )

        optimizer = keras.optimizers.SGD(lr=1e-5, momentum=0.9)
        model.compile(loss=keras.losses.Huber(), optimizer=optimizer, metrics=["mae"])
        history = model.fit(
            train_dataset,
            epochs=100,
            callbacks=[WandbCallback()],
            workers=10,
            use_multiprocessing=True,
        )

        rnn_forecast = model_forecast(
            model, series[..., np.newaxis], config.window_size
        )
        rnn_forecast = rnn_forecast[split_time - config.window_size : -1, -1, 0]
        forecast__numpy = keras.metrics.mean_absolute_error(
            x_valid, rnn_forecast
        ).numpy()
        print(forecast__numpy)

        run.summary["mean_absolute_error"] = forecast__numpy

        # save trained model as artifact
        trained_model_artifact = wandb.Artifact(
            MODEL_NAME,
            type="model",
            description="forecaster for sunspots",
            metadata=dict(config),
        )

        model.save(SAVE_MODEL_DIR)  # save using Keras
        trained_model_artifact.add_dir(SAVE_MODEL_DIR)
        run.log_artifact(trained_model_artifact)

    api = wandb.Api()
    sweep = api.sweep(
        "goofygc316/DeepLearning.AI-TensorFlow-Developer-TimeSeries/viatr2ae"
    )
    runs = sorted(sweep.runs, key=lambda run: run.summary.get("mean_absolute_error", 0))
    best_run = runs[2]
    mae = best_run.summary.get("mean_absolute_error", 0)
    print(f"Best run {best_run.name} with {mae} mean_absolute_error")
