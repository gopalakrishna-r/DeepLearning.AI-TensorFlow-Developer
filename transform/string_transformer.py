from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiment_utils import (
    create_baseline_model,
    run_experiment,
)
from preprocessor import fetch_dataset_from_csv

plt.style.use("seaborn-whitegrid")
plt.rc("figure", autolayout=True)
plt.rc(
    "axes",
    labelweight="bold",
    labelsize="large",
    titleweight="bold",
    titlesize=14,
    titlepad=10,
)

data_url = (
    "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
)
dataframe = pd.read_csv(data_url, header=None)

soil_type_values = [f"soil_type_{idx + 1}" for idx in range(40)]
wilderness_area_values = [f"area_type_{idx + 1}" for idx in range(4)]

soil_type = dataframe.loc[:, 14:53].apply(
    lambda x: soil_type_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
)
wilderness_area = dataframe.loc[:, 10:13].apply(
    lambda x: wilderness_area_values[0::1][x.to_numpy().nonzero()[0][0]], axis=1
)

CSV_HEADER = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
    "Wilderness_Area",
    "Soil_Type",
    "Cover_Type",
]

data = pd.concat(
    [dataframe.loc[:, 0:9], wilderness_area, soil_type, dataframe.loc[:, 54]],
    axis=1,
    ignore_index=True,
)
data.columns = CSV_HEADER

data["Cover_Type"] = data["Cover_Type"] - 1

train_splits, test_splits = [], []
for _, group_data in data.groupby("Cover_Type"):
    random_selection = np.random.rand(len(group_data.index)) <= 0.85
    train_splits.append(group_data[random_selection])
    test_splits.append(group_data[~random_selection])

train_data = pd.concat(train_splits).sample(frac=1).reset_index(drop=True)
test_data = pd.concat(test_splits).sample(frac=1).reset_index(drop=True)

print(f"Train split size: {len(train_data.index)}")
print(f"Test split size: {len(test_data.index)}")

TRAIN_CSV_PATH = str(Path.cwd().joinpath("dataset/train_data.csv").resolve())
TEST_CSV_PATH = str(Path.cwd().joinpath("dataset/test_data.csv").resolve())

train_data.to_csv(TRAIN_CSV_PATH, index=False)
test_data.to_csv(TEST_CSV_PATH, index=False)

TARGET_FEATURE_NAME = "Cover_Type"

TARGET_FEATURE_LABELS = ["0", "1", "2", "3", "4", "5", "6"]

NUMERIC_FEATURE_NAMES = [
    "Aspect",
    "Elevation",
    "Hillshade_3pm",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Horizontal_Distance_To_Fire_Points",
    "Horizontal_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Slope",
    "Vertical_Distance_To_Hydrology",
]

CATEGORICAL_FEATURES_WITH_VOCABULARY = {
    "Soil_Type": list(data["Soil_Type"].unique()),
    "Wilderness_Area": list(data["Wilderness_Area"].unique()),
}

COLUMN_DEFAULTS = [
    [0] if feature_name in NUMERIC_FEATURE_NAMES + [TARGET_FEATURE_NAME] else ["NA"]
    for feature_name in CSV_HEADER
]

NUM_CLASSES = len(TARGET_FEATURE_LABELS)

CSV_FEATURES = {
    "NUMERIC_FEATURE_NAMES": NUMERIC_FEATURE_NAMES,
    "CATEGORICAL_FEATURES_WITH_VOCABULARY": CATEGORICAL_FEATURES_WITH_VOCABULARY,
    "TARGET_FEATURE_NAME": TARGET_FEATURE_NAME,
}

train_dataset = fetch_dataset_from_csv(
    csv_filepath=TRAIN_CSV_PATH,
    batch_size=265,
    csv_headers=CSV_HEADER,
    col_defaults=COLUMN_DEFAULTS,
    label_names=TARGET_FEATURE_NAME,
    shuffle=True,
)
test_dataset = fetch_dataset_from_csv(
    csv_filepath=TEST_CSV_PATH,
    batch_size=265,
    csv_headers=CSV_HEADER,
    col_defaults=COLUMN_DEFAULTS,
    label_names=TARGET_FEATURE_NAME,
)

dropout_rate = 0.1
hidden_units = [32, 32]
baseline_model = create_baseline_model(
    hidden_units=hidden_units,
    num_classes=NUM_CLASSES,
    dropout=dropout_rate,
    csv_feature_names=CSV_FEATURES,
)

# keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

run_experiment(baseline_model, train_dataset, test_dataset)
