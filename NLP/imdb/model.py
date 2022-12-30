import keras as keras


def build_embedding_model(vocab_size, embedding_size, max_length=None):
    return keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embedding_size, input_length=max_length),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(6, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_text_embedding_model(vocab_size, embedding_size, dense_net_size):
    return keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embedding_size),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(dense_net_size, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_bi_embedding_model(vocab_size, embedding_size, lstm_size, dense_net_size):
    return keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embedding_size),
            keras.layers.Bidirectional(keras.layers.LSTM(lstm_size)),
            keras.layers.Dense(dense_net_size, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )


def build_stacked_bi_embedding_model(
    vocab_size, embedding_size, lstm_size, dense_net_size
):
    return keras.Sequential(
        [
            keras.layers.Embedding(vocab_size, embedding_size),
            keras.layers.Bidirectional(
                keras.layers.LSTM(lstm_size, return_sequences=True)
            ),
            keras.layers.Bidirectional(keras.layers.LSTM(32)),
            keras.layers.Dense(dense_net_size, activation="relu"),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
