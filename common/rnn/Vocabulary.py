from ..DataModule import DatasetModule
from tensorflow.keras.layers import StringLookup


class Vocab(DatasetModule):
    """Vocabulary for text."""

    def __init__(self):
        super().__init__()

    def build_vocab(self, vocab=None):
        self.vocabulary = vocab
        self.token_to_idx = StringLookup(vocabulary=vocab)
        self.idx_to_token = StringLookup(vocabulary=vocab, invert=True)

    def __len__(self):
        return self.token_to_idx.vocabulary_size()

    def __getitem__(self, tokens):
        try:
            return self.token_to_idx(tokens)
        except KeyError:
            return -1

    def to_indices(self, tokens):
        return self.__getitem__(tokens=tokens)

    def to_tokens(self, indices):
        return self.idx_to_token(indices)

    @property
    def unk(self):  # Index for the unknown token
        return self.token_to_idx["<unk>"]
