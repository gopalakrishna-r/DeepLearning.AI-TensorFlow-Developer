import tensorflow as tf


def batch_dataset(
    series, window_size, batch_size, shuffle_buffer_size, window_shift=1, cache=False
):
    series = (tf.expand_dims(series, axis=-1),)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))

    if cache:
        dataset = dataset.cache
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
        lambda batch: (batch[:-1], batch[-1:]), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def batch_csv_dataset(
    series,
    window_size,
    batch_size,
    shuffle_buffer_size=1000,
    window_shift=1,
    cache=True,
):
    series = tf.expand_dims(series, axis=-1)
    dataset = tf.data.Dataset.from_tensor_slices(series)
    dataset = dataset.window(window_size + 1, window_shift, drop_remainder=True)
    dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(
        lambda batch: (batch[:-1], batch[1:]), num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def model_forecast(model, series, window_size):
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda window: window.batch(window_size + 1))
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    forecast = model.predict(ds)
    return forecast
