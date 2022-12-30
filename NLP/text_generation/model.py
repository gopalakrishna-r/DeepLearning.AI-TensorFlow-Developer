import keras as keras
import tf_slim as slim


def build_bi_embedding_model(vocab_size, embedding_size, max_sequence_len, lstm_size):
    return keras.Sequential(
        [
            keras.layers.Embedding(
                vocab_size, embedding_size, input_length=max_sequence_len - 1
            ),
            keras.layers.Bidirectional(keras.layers.LSTM(lstm_size)),
            keras.layers.Dense(vocab_size, activation="softmax"),
        ]
    )


def build_tokenizer(shakespeare_text):
    tokenizer = keras.preprocessing.text.Tokenizer(char_level=True)
    tokenizer.fit_on_texts(shakespeare_text)
    return tokenizer


@slim.add_arg_scope
def gru_layer(*args, **kwargs):
    return keras.layers.GRU(*args, **kwargs)


def build_time_distributed_model(max_id):
    with slim.arg_scope(
        [gru_layer], return_sequences=True, dropout=0.2, recurrent_dropout=0.2
    ):
        return keras.models.Sequential(
            [
                gru_layer(128, input_shape=[None, max_id]),
                gru_layer(128),
                keras.layers.TimeDistributed(
                    keras.layers.Dense(max_id, activation="softmax")
                ),
            ]
        )


def build_stateful_time_distributed_model(max_id, batch_size=32):
    with slim.arg_scope(
        [gru_layer],
        return_sequences=True,
        dropout=0.2,
        recurrent_dropout=0.2,
        stateful=True,
    ):
        return keras.models.Sequential(
            [
                gru_layer(128, batch_input_shape=[batch_size, None, max_id]),
                gru_layer(128),
                keras.layers.TimeDistributed(
                    keras.layers.Dense(max_id, activation="softmax")
                ),
            ]
        )


class ResetStatesCallback(keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.model.reset_states()
