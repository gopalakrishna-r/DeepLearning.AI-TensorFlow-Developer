import tensorflow as tf


class MeanAbsoluteErrorWithLog(tf.keras.metrics.Metric):
    def __init__(self, name="mean_absolute_error_with_log", **kwargs):
        super().__init__(name=name, **kwargs)
        self.sum_absolute_error = self.add_weight(
            "sum_absolute_error", initializer="zeros"
        )
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_pred = tf.cast(y_pred, dtype=tf.float32)

        absolute_errors = tf.abs(tf.math.log1p(y_true) - tf.math.log1p(y_pred))

        self.sum_absolute_error.assign_add(tf.reduce_sum(absolute_errors))
        self.count.assign_add(tf.cast(tf.shape(y_true)[0], dtype=tf.float32))

    def result(self):
        return self.sum_absolute_error / self.count

    def reset_states(self):
        self.sum_absolute_error.assign(0)
        self.count.assign(0)
