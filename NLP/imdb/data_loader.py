import tensorflow_datasets as tfds


def load_imdb_data(dataset_name):
    imdb, info = tfds.load(dataset_name, with_info=True, as_supervised=True)
    train, test = imdb["train"], imdb["test"]
    return train, test, info
