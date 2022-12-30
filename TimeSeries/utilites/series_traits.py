import numpy as np


def trend(time, slope=0):
    return slope * time


def seasonal_pattern(season_time):
    return np.where(
        season_time < 0.4, np.cos(season_time * 2 * np.pi), 1 / np.exp(3 * season_time)
    )


def seasonality(time, period, amplitude=1, phase=0):
    season_time = ((time + phase) % period) / period
    return amplitude * seasonal_pattern(season_time)


def noise(time, noise_level=1):
    return np.random.randn(len(time)) * noise_level


def white_noise(time, noise_level, seed=42):
    rnd = np.random.RandomState(seed)
    return rnd.randn(len(time)) * noise_level


def autocorrelation(time, amplitude, seed=42):
    rnd = np.random.RandomState(seed)
    rho1 = 0.5
    rho2 = -0.1
    ar = rnd.randn(len(time) + 50)
    ar[:50] = 100
    for step in range(50, len(time) + 50):
        ar[step] += rho1 * ar[step - 50]
        ar[step] += rho2 * ar[step - 33]
    return ar[50:] * amplitude


def autocorrelation_with_rhos(time, amplitude, seed=42):
    rnd = np.random.RandomState(seed)
    rho = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += rho * ar[step - 1]
    return ar[1:] * amplitude


def autocorrelation(source, rhos):
    ar = source.copy()
    for step, _ in enumerate(source):
        for lag, rho in rhos.items():
            if step - lag > 0:
                ar[step] += rho * ar[step - lag]
    return ar


def impulses(time, amplitude=1, seed=None):
    rnd = np.random.RandomState(seed)
    impulse_indices = rnd.randint(len(time), size=10)
    series = np.zeros(len(time))
    for index in impulse_indices:
        series[index] += amplitude * rnd.rand()
    return series


def moving_average_forecast(series, window_size):
    forecast = [
        series[index : index + window_size].mean()
        for index in range(len(series) - window_size)
    ]
    return np.array(forecast)
