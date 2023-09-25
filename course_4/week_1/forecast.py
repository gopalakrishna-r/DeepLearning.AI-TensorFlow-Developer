# %%
# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# %% [markdown]
# <a href="https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%201%20-%20Lesson%203%20-%20Notebook.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
import tensorflow as tf

print(tf.__version__)


# %% [markdown]
# The next code block will set up the time series with seasonality, trend and a bit of noise.

# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from pathlib import Path
from datetime import datetime
import io


# %%


def rm_tree(pth):
    pth = Path(pth)
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


if (logdir := Path.joinpath(Path.cwd(), Path("logs"))).exists():
    rm_tree(str(logdir.resolve()))

Path.mkdir(logdir, parents=True, exist_ok=True)

# Clear out prior logging data.
if (plotdir := Path.joinpath(logdir, "plots")).exists():
    rm_tree(str(plotdir.resolve()))

plotdir = Path.joinpath(plotdir, f"{datetime.now().strftime('%Y%m%d-%H%M%S')}")
file_writer = tf.summary.create_file_writer(str(plotdir.resolve()))


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_series(time, series, format="-", start=0, end=None, series_name=""):
    figure = plt.figure(figsize=(10, 10))
    plt.plot(time[start:end], series[start:end], format)
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.grid(True)
    with file_writer.as_default():
        tf.summary.image(series_name, plot_to_image(figure), step=0)


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    """Just an arbitrary pattern, you can change it if you wish"""
    return np.where(
        season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    """Repeats the same pattern at each period"""
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1, seed=None):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


# %%

time = np.arange(4 * 365 + 1, dtype="float32")
baseline = 10
series = trend(time, 0.1)
baseline = 10
amplitude = 40
slope = 0.05
noise_level = 5

# Create the series
series = (
    baseline + trend(time, slope) + seasonality(time, period=365, amplitude=amplitude)
)
# Update with noise
series += noise(time, noise_level, seed=42)

plot_series(time, series, series_name="series_with_noise_trend")

# %% [markdown]
# Now that we have the time series, let's split it so we can start forecasting

# %%
split_time = 1000
time_train = time[:split_time]
x_train = series[:split_time]
time_valid = time[split_time:]
x_valid = series[split_time:]

plot_series(time_train, x_train, series_name="series_with_noise_trend_with_split")

plot_series(time_valid, x_valid, series_name="series_with_validation")


# %% [markdown]
# # Naive Forecast

# %%
naive_forecast = series[split_time - 1 : -1]

# %%
format = "-"

# %%
start = 0
end = None
figure = plt.figure(figsize=(10, 6))
plt.plot(time_valid[start:end], x_valid[start:end], format)
plt.plot(time_valid[start:end], naive_forecast[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("naive_forecast", plot_to_image(figure), step=0)

# %% [markdown]
# Let's zoom in on the start of the validation period:

# %%
figure = plt.figure(figsize=(10, 6))
start = 0
end = 150
plt.plot(time_valid[start:end], x_valid[start:end], format)
start = 1
end = 151
plt.plot(time_valid[start:end], naive_forecast[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("naive_forecast_zoomed_in", plot_to_image(figure), step=0)

# %% [markdown]
# You can see that the naive forecast lags 1 step behind the time series.

# %% [markdown]
# Now let's compute the mean squared error and the mean absolute error between the forecasts and the predictions in the validation period:

# %%
print(keras.metrics.mean_squared_error(x_valid, naive_forecast).numpy())
print(keras.metrics.mean_absolute_error(x_valid, naive_forecast).numpy())

# %% [markdown]
# That's our baseline, now let's try a moving average:


# %%
def moving_average_forecast(series, window_size):
    """Forecasts the mean of the last few values.
    If window_size=1, then this is equivalent to naive forecast"""
    forecast = [
        series[time : time + window_size].mean()
        for time in range(len(series) - window_size)
    ]
    return np.array(forecast)


# %%
moving_avg = moving_average_forecast(series, 30)[split_time - 30 :]

figure = plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid[start:end], x_valid[start:end], format)
plt.plot(time_valid[start:end], moving_avg[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("naive_forecast_moving_averages", plot_to_image(figure), step=0)

# %%
print(keras.metrics.mean_squared_error(x_valid, moving_avg).numpy())
print(keras.metrics.mean_absolute_error(x_valid, moving_avg).numpy())

# %% [markdown]
# That's worse than naive forecast! The moving average does not anticipate trend or seasonality, so let's try to remove them by using differencing. Since the seasonality period is 365 days, we will subtract the value at time *t* – 365 from the value at time *t*.

# %%
diff_series = series[365:] - series[:-365]
print(len(series[365:]), len(series[:-365]), len(diff_series), len(series))
diff_time = time[365:]

plot_series(diff_time, diff_series, series_name="seasonality_removed")

# %% [markdown]
# Great, the trend and seasonality seem to be gone, so now we can use the moving average:

# %%
diff_moving_avg = moving_average_forecast(diff_series, 50)[split_time - 365 - 50 :]

figure = plt.figure(figsize=(10, 6))
start = 0
end = None
split_diff = diff_series[split_time - 365 :]
plt.plot(time_valid[start:end], split_diff[start:end], format)
plt.plot(time_valid[start:end], diff_moving_avg[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("moving_avg_no_trend", plot_to_image(figure), step=0)

# %% [markdown]
# Now let's bring back the trend and seasonality by adding the past values from t – 365:

# %%
diff_moving_avg_plus_past = series[split_time - 365 : -365] + diff_moving_avg

figure = plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid[start:end], x_valid[start:end], format)
plt.plot(time_valid[start:end], diff_moving_avg_plus_past[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("moving_avg_with_trend", plot_to_image(figure), step=0)

# %%
print(keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_past).numpy())
print(keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_past).numpy())

# %% [markdown]
# Better than naive forecast, good. However the forecasts look a bit too random, because we're just adding past values, which were noisy. Let's use a moving averaging on past values to remove some of the noise:

# %%
diff_moving_avg_plus_smooth_past = (
    moving_average_forecast(series[split_time - 370 : -360], 10) + diff_moving_avg
)

figure = plt.figure(figsize=(10, 6))
start = 0
end = None
plt.plot(time_valid[start:end], x_valid[start:end], format)
plt.plot(time_valid[start:end], diff_moving_avg_plus_smooth_past[start:end], format)
plt.xlabel("Time")
plt.ylabel("Value")
plt.grid(True)
with file_writer.as_default():
    tf.summary.image("moving_avg_with_smoothened_past", plot_to_image(figure), step=0)

# %%
print(
    keras.metrics.mean_squared_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()
)
print(
    keras.metrics.mean_absolute_error(x_valid, diff_moving_avg_plus_smooth_past).numpy()
)

# %%
