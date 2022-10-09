import argparse

import tensorflow.keras as keras
import tensorflow_hub as hub
import tensorflow as tf
import numpy as np
import os
import tensorflow_datasets as tfds

from NLP.sarcasm.visualizer import plot_graphs
from data_loader import load_imdb_data
from model import build_embedding_model, build_text_embedding_model, build_bi_embedding_model, \
    build_stacked_bi_embedding_model
from preprocessor import process_sequence, build_vocab, build_table, pre_process
from visualizer import write_embeddings, reverse_word_index

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('rnn_model', help='comma separated list of models to use.', default='1')
    args = parser.parse_args()
    if args.rnn_model:
        rnn_model = int(args.rnn_model)
        if rnn_model == 1:
            rnn_model_type = int(args.rnn_type)

            train_data, test_data, _ = load_imdb_data("imdb_reviews")
            vocab_size = 10000
            max_length = 120

            training_sentences, training_labels, testing_sentences, testing_labels = [], [], [], []

            for s, l in train_data:
                training_sentences.append(s.numpy().decode('utf-8'))
                training_labels.append(l.numpy())

            for s, l in test_data:
                testing_sentences.append(s.numpy().decode('utf-8'))
                testing_labels.append(l.numpy())

            training_labels_final = np.array(training_labels)
            testing_labels_final = np.array(testing_labels)

            padded, padded_testing, word_index = process_sequence(training_sentences, testing_sentences, vocab_size,
                                                                  max_length)
            model = build_embedding_model(vocab_size, 16, max_length)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print(model.summary())

            num_epochs = 10
            model.fit(padded, training_labels_final, validation_data=(padded_testing, testing_labels_final),
                      epochs=num_epochs,
                      workers=8)

            e = model.layers[0]
            weights = e.get_weights()[0]
            print(weights.shape)

            write_embeddings(weights, vocab_size, lambda word_ind: reverse_word_index(word_ind))

        elif rnn_model == 2:
            rnn_model_type = int(args.rnn_type)
            train_data, test_data, info = load_imdb_data("imdb_reviews/subwords8k")
            tokenizer = info.features['text'].encoder
            print(tokenizer.subwords)

            sample_string = 'Tensorflow, from basics to mastery'
            toknized_string = tokenizer.encode(sample_string)

            for ts in toknized_string:
                print(f'{ts} ----> {tokenizer.decode([ts])}')

            BUFFER_SIZE = 10000
            BATCH_SIZE = 64
            model = None
            if rnn_model_type == 1:
                embedding_size = 16
                model = build_text_embedding_model(tokenizer.vocab_size, embedding_size, 6)
            elif rnn_model_type == 2:
                lstm_size = 64
                embedding_size = 64
                model = build_bi_embedding_model(tokenizer.vocab_size, embedding_size, lstm_size, 64)
            elif rnn_model_type == 3:
                lstm_size = 64
                embedding_size = 64
                model = build_stacked_bi_embedding_model(tokenizer.vocab_size, embedding_size, lstm_size, 64)

            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print(model.summary())

            num_epochs = 10

            train_data = train_data.shuffle(BUFFER_SIZE)
            train_data = train_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(train_data))
            test_data = test_data.padded_batch(BATCH_SIZE, tf.compat.v1.data.get_output_shapes(test_data))

            history = model.fit(train_data, validation_data=test_data, epochs=num_epochs, workers=8)

            plot_graphs(history, 'acc')
            plot_graphs(history, 'loss')
            # e = model.layers[0]
            # weights = e.get_weights()[0]
            # print(weights.shape)
            # write_embeddings(weights, tokenizer.vocab_size, lambda word: tokenizer.decode(word))
        elif rnn_model == 3:
            train, test, info = load_imdb_data('imdb_reviews')
            train_size, test_size = info.splits['train'].num_examples, info.splits['test'].num_examples
            num_oov_buckets, vocab_size, embedding_size = 1000, 10000, 128
            vocab = build_vocab(train)

            table = build_table(vocab, num_oov_buckets, vocab_size)
            train_set = train.repeat().batch(32).map(pre_process)
            train_set = train_set.map(lambda x_batch, _: (table.lookup(x_batch), _)).prefetch(1)

            model = keras.models.Sequential([
                keras.layers.Embedding(vocab_size + num_oov_buckets, embedding_size, mask_zero=True,
                                       input_shape=[None]),
                keras.layers.GRU(128, return_sequences=True),
                keras.layers.GRU(128),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
            history = model.fit(train_set, steps_per_epoch=train_size // 32, epochs=5)

        elif rnn_model ==4:
            tf.random.set_seed(42)
            TFHUB_CACHE_DIR = os.path.join(os.curdir, "tfhub_cache")
            os.environ["TFHUB_CACHE_DIR"] = TFHUB_CACHE_DIR

            model = keras.models.Sequential([
                hub.KerasLayer("https://tfhub.dev/google/tf2-preview/nnlm-en-dim50/1", dtype=tf.string,
                               input_shape=[], output_shape=[50]),
                keras.layers.Dense(128, activation='relu'),
                keras.layers.Dense(1, activation='sigmoid')
            ])

            model.compile(optimizer='adam', metrics = ['accuracy'], loss='binary_crossentropy')

            for dirpath, _, filenames in os.walk(TFHUB_CACHE_DIR):
                for filename in filenames:
                    print(os.path.join(dirpath, filename))

            datasets, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)
            train_size = info.splits["train"].num_examples
            batch_size = 32
            train_set = datasets["train"].repeat().batch(batch_size).prefetch(1)
            history = model.fit(train_set, epochs = 10, steps_per_epoch = train_size//batch_size, workers = 16)
        else:
            print(f'Invalid run choice')
