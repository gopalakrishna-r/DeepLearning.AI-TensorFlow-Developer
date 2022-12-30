import argparse

import numpy as np

from NLP.NMT.NMTModel import NMTModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "rnn_model", help="comma separated list of models to use.", default="1"
    )
    args = parser.parse_args()
    if args.rnn_model:
        rnn_model = int(args.rnn_model)
        if rnn_model == 1:
            model = NMTModel(vocab_size=100, embedding_size=10)
            model.compile(loss="sparse_categorical_crossentropy", optimizer="adam")

            X = np.random.randint(100, size=10 * 1000).reshape(1000, 10)
            Y = np.random.randint(100, size=15 * 1000).reshape(1000, 15)
            X_decoder = np.c_[np.zeros((1000, 1)), Y[:, :-1]]
            seq_lengths = np.full([1000], 15)

            history = model.fit([X, X_decoder, seq_lengths], Y, epochs=5, workers=16)

        else:
            print(f"Invalid run choice")
