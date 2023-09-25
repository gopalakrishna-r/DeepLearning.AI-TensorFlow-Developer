import tensorflow as tf
from .CSV_KEYS import CLIMATE
from tensorflow.keras import backend
import numpy as np
from common.Visualizer import Visualizer
from course_4.week_1.time_series.models.TimeSeriesModels import Baseline, RNNForecaster
from .RNNTrainer import TimeSeriesTrainer
from .RNNSeries import RNNSeriesData
import matplotlib.pyplot as plt

if __name__ == "__main__":
    log_file = "/mnt/d/codebase/deepai_specialization/out.txt"
    backend.clear_session()
    visualizer = Visualizer()
    ds_batch_size = 4

    w1 = RNNSeriesData(
        visualizer=visualizer,
        input_width=24,
        label_width=1,
        shift=24,
        label_columns=["T (degC)"],
        batch_size=1,
    )

    tf.print(w1, output_stream=f"file://{log_file}")

    w2 = RNNSeriesData(
        visualizer=visualizer,
        input_width=6,
        label_width=1,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=1,
    )
    tf.print(w2, output_stream=f"file://{log_file}")
    baseline_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)

    example_window = tf.stack(
        [
            np.array(w2.train_df[: w2.total_window_size]).astype(np.float32),
            np.array(w2.train_df[100 : 100 + w2.total_window_size]).astype(np.float32),
            np.array(w2.train_df[200 : 200 + w2.total_window_size]).astype(np.float32),
        ]
    )

    example_inputs, example_labels = w2.example

    tf.print(
        "All shapes are: (batch, time, features)", output_stream=f"file://{log_file}"
    )
    tf.print(
        f"Window shape: {example_window.shape}", output_stream=f"file://{log_file}"
    )
    tf.print(
        f"Inputs shape: {example_inputs.shape}", output_stream=f"file://{log_file}"
    )
    tf.print(
        f"Labels shape: {example_labels.shape}", output_stream=f"file://{log_file}"
    )

    w2.example = example_inputs, example_labels
    w2.plot()
    w2.plot(plot_col="p (mbar)")
    tf.print(
        f"w2 element shape: {w2.train.element_spec}", output_stream=f"file://{log_file}"
    )

    for example_inputs, example_labels in w2.train.take(1):
        tf.print(
            f"Inputs shape (batch, time, features): {example_inputs.shape}",
            output_stream=f"file://{log_file}",
        )
        tf.print(
            f"Labels shape (batch, time, features): {example_labels.shape}",
            output_stream=f"file://{log_file}",
        )

    single_step_window = RNNSeriesData(
        visualizer=visualizer,
        input_width=1,
        label_width=1,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=ds_batch_size,
    )
    tf.print(single_step_window, output_stream=f"file://{log_file}")
    for example_inputs, example_labels in single_step_window.train.take(1):
        tf.print(
            f"Inputs shape (batch, time, features): {example_inputs.shape}",
            output_stream=f"file://{log_file}",
        )
        tf.print(
            f"Labels shape (batch, time, features): {example_labels.shape}",
            output_stream=f"file://{log_file}",
        )

    baseline = Baseline(label_index=CLIMATE.COLUMN_KEYS["T (degC)"])
    baseline_forecaster = RNNForecaster(rnn=baseline)
    baseline_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    baseline_forecaster.compile(
        loss=baseline_forecaster.configure_loss(),
        metrics=[tf.keras.metrics.MeanAbsoluteError()],
    )

    val_performance = {}
    performance = {}
    val_performance["Baseline"] = baseline_forecaster.evaluate(single_step_window.valid)
    performance["Baseline"] = baseline_forecaster.evaluate(single_step_window.test)

    wide_window = RNNSeriesData(
        visualizer=visualizer,
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=ds_batch_size,
    )

    tf.print(wide_window, output_stream=f"file://{log_file}")
    tf.print(
        "Input shape:", wide_window.example[0].shape, output_stream=f"file://{log_file}"
    )
    tf.print(
        "Output shape:",
        baseline(wide_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )
    wide_window.plot(baseline_forecaster, plot_title="baseline_preds")

    forecaster = RNNForecaster(
        rnn=tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    )

    tf.print(
        "Input shape:",
        single_step_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        forecaster(single_step_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    trainer.fit(forecaster, single_step_window)

    val_performance["Linear"] = trainer.evaluate(single_step_window.valid)
    performance["Linear"] = trainer.evaluate(single_step_window.test)
    tf.print(
        "Input shape:", wide_window.example[0].shape, output_stream=f"file://{log_file}"
    )
    tf.print(
        "Output shape:",
        forecaster(wide_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    wide_window.plot(forecaster, plot_title="linear_preds")

    dense_forecaster = RNNForecaster(
        rnn=tf.keras.Sequential(
            [
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=64, activation="relu"),
                tf.keras.layers.Dense(units=1),
            ]
        )
    )
    dense_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    dense_trainer.fit(forecaster, single_step_window)

    val_performance["Dense"] = dense_trainer.evaluate(single_step_window.valid)
    performance["Dense"] = dense_trainer.evaluate(single_step_window.test)

    CONV_WIDTH = 3
    conv_window = RNNSeriesData(
        visualizer=visualizer,
        input_width=CONV_WIDTH,
        label_width=1,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=ds_batch_size,
    )
    tf.print(
        "conv_window:",
        conv_window,
        output_stream=f"file://{log_file}",
    )

    conv_window.plot(plot_title="3_hour_preds")

    multi_dense_forecaster = RNNForecaster(
        rnn=tf.keras.Sequential(
            [
                # Shape: (time, features) => (time*features)
                tf.keras.layers.Flatten(),
                tf.keras.layers.Dense(units=32, activation="relu"),
                tf.keras.layers.Dense(units=32, activation="relu"),
                tf.keras.layers.Dense(units=1),
                # Add back the time dimension.
                # Shape: (outputs) => (1, outputs)
                tf.keras.layers.Reshape([1, -1]),
            ]
        )
    )

    tf.print(
        "Input shape:",
        conv_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        multi_dense_forecaster(conv_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    multi_dense_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    multi_dense_trainer.fit(multi_dense_forecaster, conv_window)

    val_performance["multi_dense"] = multi_dense_trainer.evaluate(conv_window.valid)
    performance["multi_dense"] = multi_dense_trainer.evaluate(conv_window.test)

    conv_window.plot(multi_dense_forecaster)

    tf.print(
        "Input shape:",
        wide_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    try:
        tf.print(
            "Output shape:",
            multi_dense_forecaster(wide_window.example[0]).shape,
            output_stream=f"file://{log_file}",
        )
    except Exception as e:
        tf.print(
            f"\n{type(e).__name__}:{e}",
            output_stream=f"file://{log_file}",
        )

    conv_model_forecaster = RNNForecaster(
        rnn=tf.keras.Sequential(
            [
                tf.keras.layers.Conv1D(
                    filters=32, kernel_size=(CONV_WIDTH,), activation="relu"
                ),
                tf.keras.layers.Dense(units=32, activation="relu"),
                tf.keras.layers.Dense(units=1),
            ]
        )
    )
    tf.print(
        "Conv model on `conv_window`",
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Input shape:",
        conv_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        conv_model_forecaster(conv_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    conv_dense_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    conv_dense_trainer.fit(conv_model_forecaster, conv_window)

    val_performance["Conv"] = conv_dense_trainer.evaluate(conv_window.valid)
    performance["Conv"] = conv_dense_trainer.evaluate(conv_window.test)

    tf.print(
        "Conv model on `wide_window`",
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Input shape:",
        wide_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        conv_model_forecaster(wide_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = RNNSeriesData(
        visualizer=visualizer,
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=ds_batch_size,
    )

    tf.print(
        "Wide conv window",
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Input shape:",
        wide_conv_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        conv_model_forecaster(wide_conv_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    wide_conv_window.plot(conv_model_forecaster, plot_title="wide_conv_window")

    lstm_forecaster = RNNForecaster(
        rnn=tf.keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=1),
            ]
        )
    )
    tf.print(
        "Wide window on lstm",
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Input shape:",
        wide_window.example[0].shape,
        output_stream=f"file://{log_file}",
    )
    tf.print(
        "Output shape:",
        lstm_forecaster(wide_window.example[0]).shape,
        output_stream=f"file://{log_file}",
    )

    lstm_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    lstm_trainer.fit(lstm_forecaster, wide_window)

    val_performance["LSTM"] = lstm_trainer.evaluate(wide_window.valid)
    performance["LSTM"] = lstm_trainer.evaluate(wide_window.test)

    wide_window.plot(lstm_forecaster, plot_title="lstm_window")

    x = np.arange(len(performance))
    width = 0.3
    metric_name = "mean_absolute_error"
    metric_index = lstm_trainer.model.metrics_names.index("mean_absolute_error")
    val_mae = list(map(lambda _: _[metric_index], val_performance.values()))
    test_mae = list(map(lambda _: _[metric_index], performance.values()))

    figure = plt.figure(figsize=(12, 8))
    plt.ylabel("mean_absolute_error [T (degC), normalized]")
    plt.bar(x - 0.17, val_mae, width, label="Validation")
    plt.bar(x + 0.17, test_mae, width, label="Test")
    plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
    _ = plt.legend()
    with visualizer.file_writer.as_default():
        tf.summary.image(
            "performance_metrics", visualizer.plot_to_image(figure), step=0
        )

    for name, value in performance.items():
        tf.print(f"{name:12s}: {value[1]:0.4f}", output_stream=f"file://{log_file}")

    wide_window = RNNSeriesData(
        visualizer=visualizer,
        input_width=24,
        label_width=24,
        shift=1,
        label_columns=["T (degC)"],
        batch_size=ds_batch_size,
    )
    num_features = wide_window.slice_df.shape[1]
    multi_lstm_forecaster = RNNForecaster(
        rnn=tf.keras.models.Sequential(
            [
                # Shape [batch, time, features] => [batch, time, lstm_units]
                tf.keras.layers.LSTM(32, return_sequences=True),
                # Shape => [batch, time, features]
                tf.keras.layers.Dense(units=num_features),
            ]
        )
    )

    multi_lstm_trainer = TimeSeriesTrainer(visualizer=visualizer, max_epochs=20)
    multi_lstm_trainer.fit(multi_lstm_forecaster, wide_window)

    val_performance["LSTM"] = multi_lstm_forecaster.evaluate(wide_window.valid)
    performance["LSTM"] = multi_lstm_forecaster.evaluate(wide_window.test)

    wide_window.plot(multi_lstm_forecaster, plot_title="multi_lstm_output")
