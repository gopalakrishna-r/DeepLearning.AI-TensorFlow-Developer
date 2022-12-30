import argparse

import numpy as np
import tensorflow.keras as keras
from functional import seq
from tensorflow.keras.preprocessing.sequence import pad_sequences

from NLP.text_generation.visualizer import plot_graphs
from data_loader import (
    load_corpus_data,
    load_irish_songs,
    load_shakespeare_text,
    generate_shakespeare_dataset,
    generate_shakespeare_dataset_stateful,
)
from model import (
    build_bi_embedding_model,
    build_tokenizer,
    build_time_distributed_model,
    build_stateful_time_distributed_model,
    ResetStatesCallback,
)
from preprocessor import process_sequence

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rnn_model", help="comma separated list of models to use.", default="1"
    )
    args = parser.parse_args()
    if args.rnn_model:
        rnn_model = int(args.rnn_model)
        if rnn_model == 1:
            xs, ys, total_words, max_sequence_len, tokenizer = process_sequence(
                load_corpus_data()
            )

            model = build_bi_embedding_model(total_words, 64, max_sequence_len, 20)
            model.compile(
                loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
            )
            print(model.summary())

            num_epochs = 500
            history = model.fit(xs, ys, epochs=num_epochs, workers=8)

            seed_text = "Laurence went to dublin"
            next_words = 100

            for _ in range(next_words):
                output_word = (
                    seq([tokenizer.texts_to_sequences([seed_text])[0]])
                    .map(
                        lambda token: pad_sequences(
                            [token], maxlen=max_sequence_len - 1, padding="pre"
                        )
                    )
                    .map(lambda token: np.argmax(model.predict(token), axis=-1))
                    .map(
                        lambda predicted_word: (
                            seq(tokenizer.word_index.items()).find(
                                lambda key_index_tuple: key_index_tuple[1]
                                == predicted_word
                            )
                        )
                    )
                    .map(
                        lambda key_index_tuple: key_index_tuple[0]
                        if key_index_tuple
                        else ""
                    )
                    .fold_left("", lambda current, next_word: current + " " + next_word)
                )
                seed_text += output_word
            print(seed_text)
        elif rnn_model == 2:
            xs, ys, total_words, max_sequence_len, tokenizer = process_sequence(
                load_irish_songs()
            )

            model = build_bi_embedding_model(total_words, 100, max_sequence_len, 150)
            model.compile(
                loss="categorical_crossentropy",
                optimizer=keras.optimizers.Adam(learning_rate=0.01),
                metrics=["accuracy"],
            )
            num_epochs = 100
            history = model.fit(xs, ys, epochs=num_epochs, workers=8)
            plot_graphs(history, "accuracy")

            seed_text = "help me, obi wan kenobi. you are my only hope"
            next_words = 100

            for _ in range(next_words):
                output_word = (
                    seq([tokenizer.texts_to_sequences([seed_text])[0]])
                    .map(
                        lambda token: pad_sequences(
                            [token], maxlen=max_sequence_len - 1, padding="pre"
                        )
                    )
                    .map(lambda token: np.argmax(model.predict(token), axis=-1))
                    .map(
                        lambda predicted_word: (
                            seq(tokenizer.word_index.items()).find(
                                lambda key_index_tuple: key_index_tuple[1]
                                == predicted_word
                            )
                        )
                    )
                    .map(
                        lambda key_index_tuple: key_index_tuple[0]
                        if key_index_tuple
                        else ""
                    )
                    .fold_left("", lambda current, next_word: current + " " + next_word)
                )
                seed_text += output_word
            print(seed_text)
        elif rnn_model == 3:
            shakespeare_txt = load_shakespeare_text()

            tokenizer = build_tokenizer(shakespeare_txt)
            max_id = len(tokenizer.word_index)
            dataset_size = tokenizer.document_count

            [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_txt])) - 1
            dataset = generate_shakespeare_dataset(encoded, dataset_size, max_id)

            for X_batch, Y_batch in dataset.take(1):
                print(X_batch.shape, Y_batch.shape)

            model = build_time_distributed_model(max_id)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            print(model.summary())
            history = model.fit(dataset, epochs=2, workers=8, use_multiprocessing=True)
            model.save("char-rnn-model.h5")
        elif rnn_model == 4:
            shakespeare_txt = load_shakespeare_text()

            tokenizer = build_tokenizer(shakespeare_txt)
            max_id = len(tokenizer.word_index)
            dataset_size = tokenizer.document_count

            [encoded] = np.array(tokenizer.texts_to_sequences([shakespeare_txt])) - 1
            dataset = generate_shakespeare_dataset_stateful(
                encoded, dataset_size, max_id
            )

            for X_batch, Y_batch in dataset.take(1):
                print(X_batch.shape, Y_batch.shape)

            model = build_stateful_time_distributed_model(max_id)
            model.compile(
                optimizer="adam",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"],
            )
            print(model.summary())
            history = model.fit(
                dataset,
                epochs=50,
                workers=8,
                use_multiprocessing=True,
                callbacks=[ResetStatesCallback()],
            )
