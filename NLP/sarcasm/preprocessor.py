from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer


def process_sequence(
    training_sentences,
    testing_sentences,
    vocab_size,
    max_length,
    trunc_type,
    padding_type,
    oov_token,
):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(training_sentences)
    sequences = tokenizer.texts_to_sequences(training_sentences)
    padded = pad_sequences(
        sequences, maxlen=max_length, truncating=trunc_type, padding=padding_type
    )

    testing_seq = tokenizer.texts_to_sequences(testing_sentences)
    testing_padded = pad_sequences(
        testing_seq, maxlen=max_length, truncating=trunc_type, padding=padding_type
    )
    return padded, testing_padded
