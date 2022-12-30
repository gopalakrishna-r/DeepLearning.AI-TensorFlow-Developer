import csv
from pathlib import Path

import numpy as np
import tensorflow.keras as keras

SUNSPOTS_URI = (
    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/Sunspots.csv"
)
DATA_PATH = Path.cwd().joinpath("datasets")
SPLIT_DATA_PATH = DATA_PATH.joinpath("split")
SUNSPOTS_CSV_FILE = DATA_PATH.joinpath("Sunspots.csv")


def fetch_sunspots_data(csv_data, header, split_time):
    csv_train = csv_data[:split_time]
    csv_valid = csv_data[split_time:]
    train_filepaths = save_to_multiple_files(csv_train, header, "train", n_parts=20)
    valid_filepaths = save_to_multiple_files(csv_valid, header, "valid", n_parts=10)
    return train_filepaths, valid_filepaths


def process_csv_data():
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    keras.utils.get_file(
        origin=SUNSPOTS_URI, fname=SUNSPOTS_CSV_FILE, cache_dir=DATA_PATH
    )
    with open(SUNSPOTS_CSV_FILE) as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        header = ",".join(next(reader))
        return [*reader], header


def save_to_multiple_files(data, header, name_prefix, n_parts=10):
    SPLIT_DATA_PATH.mkdir(parents=True, exist_ok=True)
    path_format = Path.joinpath(SPLIT_DATA_PATH, "sunspots_{}_{:02d}.csv")
    count = len(data)
    filepaths = []
    for file_idx, row_indices in enumerate(np.array_split(np.arange(count), n_parts)):
        part_csv = str(path_format.resolve()).format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            f.write(header)
            f.write("\n")
            for row_idx in row_indices:
                f.write((data[row_idx][2]))
                f.write("\n")
    return filepaths
