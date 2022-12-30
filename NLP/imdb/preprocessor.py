from collections import Counter

import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def process_sequence(
    training_sentences,
    testing_sentences,
    vocab_size,
    max_length,
    trunc_type="post",
    oov_type="<OOV>",
):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_type)
    tokenizer.fit_on_texts(training_sentences)
    word_index = tokenizer.word_index
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(sequences, maxlen=max_length, truncating=trunc_type)

    testing_seq = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(testing_seq, maxlen=max_length)
    return padded, testing_padded, word_index


def pre_process(x_batch, y_batch):
    x_batch = tf.strings.substr(x_batch, 0, 300)
    x_batch = tf.strings.regex_replace(x_batch, b"<br\\s*/?>", b" ")
    x_batch = tf.strings.regex_replace(x_batch, b"[^a-zA-Z']", b" ")
    x_batch = tf.strings.split(x_batch)
    return x_batch.to_tensor(default_value=b"<pad>"), y_batch


def build_vocab(train):
    vocab = Counter()
    for x_batch, y_batch in train.batch(32).map(pre_process):
        for review in x_batch:
            vocab.update(list(review.numpy()))
    return vocab


def build_table(vocab, num_oov_buckets, vocab_size):
    truncated_vocab = [word for word, _ in vocab.most_common()[:vocab_size]]
    words = tf.constant(truncated_vocab)
    word_ids = tf.range(len(truncated_vocab), dtype=tf.int64)
    vocab_init = tf.lookup.KeyValueTensorInitializer(words, word_ids)

    return tf.lookup.StaticVocabularyTable(vocab_init, num_oov_buckets)


def encode_words(x_batch, y_batch, table):
    return table.lookup(x_batch), y_batch
