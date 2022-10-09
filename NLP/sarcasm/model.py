import tensorflow.keras as keras


def build_embedding_model(config):
    return keras.Sequential([
        keras.layers.Embedding(config.vocab_size, config.embedding_size, input_length=config.max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(24, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
