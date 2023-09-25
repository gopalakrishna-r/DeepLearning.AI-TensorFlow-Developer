import numpy as np
import tensorflow as tf


class ClimateDataProcessor(tf.keras.layers.Layer):
    def __init__(self, name="preprocess_tf", column_indices={}, **kwargs):
        super(ClimateDataProcessor, self).__init__(name=name, **kwargs)
        self.tf_pi = tf.constant(np.pi, dtype=tf.float32)
        self.day = tf.constant(24 * 60 * 60, dtype=tf.float32)
        self.year = tf.multiply(365.2425, self.day)
        self.column_indices = column_indices

    def call(self, input_x, **kwargs):
        update_truths = tf.equal(input_x, -9999.00)
        input_x = tf.cond(
            tf.reduce_any(update_truths),
            lambda: tf.where(
                update_truths,
                input_x,
                tf.constant([0.0], tf.float32),
            ),
            lambda: input_x,
        )

        wv = input_x[:, :, self.column_indices["wv (m/s)"]]
        max_wv = input_x[:, :, self.column_indices["max. wv (m/s)"]]
        # Convert to radians.
        wd_rad = tf.divide(
            tf.multiply(input_x[:, :, self.column_indices["wd (deg)"]], self.tf_pi),
            180.0,
        )

        # Calculate the wind x and y components.
        Wx = tf.multiply(wv, tf.cos(wd_rad))
        Wy = tf.multiply(wv, tf.sin(wd_rad))

        # Calculate the max wind x and y components.
        max_Wx = tf.multiply(max_wv, tf.cos(wd_rad))
        max_Wy = tf.multiply(max_wv, tf.sin(wd_rad))

        timestamp_s = tf.cast(
            input_x[:, :, self.column_indices["Time_Stamp"]], dtype=tf.float32
        )

        day_deg = tf.divide(tf.multiply(2.0, self.tf_pi), self.day)
        Day_sin = tf.sin(tf.multiply(timestamp_s, day_deg))
        Day_cos = tf.cos(tf.multiply(timestamp_s, day_deg))
        year_deg = tf.divide(tf.multiply(2.0, self.tf_pi), self.year)
        Year_sin = tf.sin(tf.multiply(timestamp_s, year_deg))
        Year_cos = tf.cos(tf.multiply(timestamp_s, year_deg))
        stacked_features = tf.stack(
            [Wx, Wy, max_Wx, max_Wy, Day_sin, Day_cos, Year_sin, Year_cos], axis=-1
        )
        return tf.concat([input_x[:, :, :11], stacked_features], axis=-1)

    def get_config(self):
        config = super(ClimateDataProcessor, self).get_config()
        return config
