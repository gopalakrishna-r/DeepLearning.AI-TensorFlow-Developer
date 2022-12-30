import itertools

import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical


def process_sequence(data):
    tokenizer = Tokenizer()
    corpus = data.lower().split("\n")
    tokenizer.fit_on_texts(corpus)

    total_words = len(tokenizer.word_index) + 1

    print(tokenizer.word_index)
    print(f"length of word_index {total_words}")

    input_sequences = list(
        itertools.chain.from_iterable(
            [
                [token_list[: i + 1] for i in range(1, len(token_list))]
                for token_list in [
                    tokenizer.texts_to_sequences([line])[0] for line in corpus
                ]
            ]
        )
    )

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(
        pad_sequences(input_sequences, maxlen=max_sequence_len, padding="pre")
    )

    xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
    ys = to_categorical(labels, num_classes=total_words)

    return xs, ys, total_words, max_sequence_len, tokenizer
