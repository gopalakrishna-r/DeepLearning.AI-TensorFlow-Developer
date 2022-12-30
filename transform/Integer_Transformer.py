from pathlib import Path
import tensorflow as tf
import keras
import numpy as np
import pandas as pd
from keras import optimizers, losses

from CSV_KEYS import titanic
from experiment_utils import (
    create_bl_model,
    build_column_defaults,
)
from preprocessor import fetch_dataset_from_csv

TRAIN_CSV_PATH = str(Path.cwd().joinpath("dataset/titanic/train.csv").resolve())
TEST_CSV_PATH = str(Path.cwd().joinpath("dataset/titanic/test.csv").resolve())
SUBMISSION_PATH = str(
    Path.cwd().joinpath("dataset/titanic/gender_submission.csv").resolve()
)
train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)
TRANSFORMED_TEST_CSV_PATH = str(
    Path.cwd().joinpath("dataset/titanic/transformed/test.csv").resolve()
)
TRANSFORMED_TRAIN_CSV_PATH = str(
    Path.cwd().joinpath("dataset/titanic/transformed/train.csv").resolve()
)
CSV_HEADER = [
    "PassengerId",
    "Survived",
    "Pclass",
    "Name",
    "Sex",
    "Age",
    "SibSp",
    "Parch",
    "Ticket",
    "Fare",
    "Cabin",
    "Embarked",
]
COLUMN_DEFAULTS = build_column_defaults(
    titanic.NUMERICAL_CAT_CONT_KEYS,
    titanic.NUMERICAL_CAT_NONCONT_KEYS,
    titanic.TARGET_FEATURE_NAME,
    csv_headers=CSV_HEADER,
)
CSV_HEADER_TEST = CSV_HEADER.copy()
CSV_HEADER_TEST.remove(titanic.TARGET_FEATURE_NAME)

COLUMN_DEFAULTS_FOR_TEST = build_column_defaults(
    titanic.NUMERICAL_CAT_CONT_KEYS,
    titanic.NUMERICAL_CAT_NONCONT_KEYS,
    titanic.TARGET_FEATURE_NAME,
    csv_headers=CSV_HEADER_TEST,
)
batch_size = 64
train_dataset = fetch_dataset_from_csv(
    csv_filepath=TRAIN_CSV_PATH,
    batch_size=batch_size,
    csv_headers=CSV_HEADER,
    col_defaults=COLUMN_DEFAULTS,
    label_names=titanic.TARGET_FEATURE_NAME,
    shuffle=True,
)
test_dataset = fetch_dataset_from_csv(
    csv_filepath=TEST_CSV_PATH,
    batch_size=batch_size,
    col_defaults=COLUMN_DEFAULTS_FOR_TEST,
)
TEXT_FEATURES_WITH_VOCABULARY = dict(
    map(
        lambda feature: (feature, list(train_csv[feature].unique())),
        titanic.TEXT_FEAT_KEYS,
    )
)
NUMERIC_FEATURES_WITH_VOCAB = dict(
    continous=dict(
        map(
            lambda feature: (feature, train_csv[feature]),
            titanic.NUMERICAL_CAT_CONT_KEYS,
        )
    ),
    noncontinous=dict(
        map(
            lambda feature: (feature, np.unique(train_csv[feature])),
            titanic.NUMERICAL_CAT_NONCONT_KEYS,
        )
    ),
)
CATEGORICAL_FEATURES_WITH_VOCABULARY = dict(
    map(
        lambda feature: (feature, list(train_csv[feature].unique())),
        titanic.CATEGORICAL_FEAT_KEYS,
    )
)
CSV_FEATURES = {
    "NUMERIC_FEATURES_WITH_VOCABULARY": NUMERIC_FEATURES_WITH_VOCAB,
    "CATEGORICAL_FEATURES_WITH_VOCABULARY": CATEGORICAL_FEATURES_WITH_VOCABULARY,
    "TEXT_FEATURE_NAMES_WITH_VOCABULARY": TEXT_FEATURES_WITH_VOCABULARY,
    "TARGET_FEATURE_NAME": titanic.TARGET_FEATURE_NAME,
}
dropout_rate = 0.1
hidden_units = [64, 64]
learning_rate = 0.01

csv_total = train_csv.append(test_csv)

children_with_no_guardian = (
    csv_total[
        (csv_total.Age.isnull())
        & (csv_total.Name.str.contains("Master"))
        & (csv_total.Parch == 0)
    ]
).PassengerId.values[0]
test_csv.loc[test_csv.PassengerId == children_with_no_guardian, "Age"] = 14

# extract ages from all the titles
train_csv["Title"], test_csv["Title"] = [
    df.Name.str.extract(" ([A-Za-z]+)\.", expand=False) for df in [train_csv, test_csv]
]

# map titles to most frequent ones
TitleDict = {
    "Capt": "Officer",
    "Col": "Officer",
    "Major": "Officer",
    "Jonkheer": "Royalty",
    "Don": "Royalty",
    "Sir": "Royalty",
    "Dr": "Royalty",
    "Rev": "Royalty",
    "Countess": "Royalty",
    "Mme": "Mrs",
    "Mlle": "Miss",
    "Ms": "Mrs",
    "Mr": "Mr",
    "Mrs": "Mrs",
    "Miss": "Miss",
    "Master": "Master",
    "Lady": "Royalty",
}

train_csv["Title"], test_csv["Title"] = [
    df.Title.map(TitleDict) for df in [train_csv, test_csv]
]

test_csv.at[414, "Title"] = "Royalty"

# fare per person

for df in [train_csv, test_csv]:
    df["PeopleInTicket"] = df["Ticket"].map(csv_total["Ticket"].value_counts())
    df["FarePerPerson"] = df["Fare"] / df["PeopleInTicket"]

test_csv.loc[test_csv.Fare.isnull(), ["Fare", "FarePerPerson"]] = round(
    train_csv[
        (train_csv.Embarked == "S")
        & (train_csv.Pclass == 3)
        & (train_csv.PeopleInTicket == 1)
    ]["Fare"].mean(),
    1,
)

print(csv_total[csv_total.Embarked.isnull()])

print(
    train_csv[(train_csv.Pclass == 1)]
    .groupby("Embarked")
    .agg({"FarePerPerson": "mean", "Fare": "mean", "PassengerId": "count"})
)

print(
    train_csv[
        (train_csv.Pclass == 1) & (train_csv.PeopleInTicket == 2) & (train_csv.Age > 18)
    ]
    .groupby("Embarked")
    .agg({"FarePerPerson": "mean", "Fare": "mean", "PassengerId": "count"})
)

train_csv.Embarked.fillna("C", inplace=True)

print(
    train_csv.groupby(["Parch", "Sex", "Title"])["Age"].agg({"mean", "median", "count"})
)

for df in [train_csv, test_csv]:
    df.loc[
        (df["Title"] == "Miss") & (df["Parch"] != 0) & (df["PeopleInTicket"] > 1),
        "Title",
    ] = "FemaleChild"

grp = (
    train_csv.groupby(["Pclass", "Sex", "Title"])["Age"]
    .mean()
    .reset_index()[["Sex", "Pclass", "Title", "Age"]]
)


def fill_age(x):
    return grp[(grp.Pclass == x.Pclass) & (grp.Sex == x.Sex) & (grp.Title == x.Title)][
        "Age"
    ].values[0]


# Here 'x' is the row containing the missing age. We look up the row's Pclass
# Sex and Title against the lookup table as shown previously and return the Age
# Now we have to call this fill_age function for every missing row for test, train

train_csv["Age"], test_csv["Age"] = [
    df.apply(lambda x: fill_age(x) if np.isnan(x["Age"]) else x["Age"], axis=1)
    for df in [train_csv, test_csv]
]

train_csv.to_csv(TRANSFORMED_TRAIN_CSV_PATH, index=False)
test_csv.to_csv(TRANSFORMED_TEST_CSV_PATH, index=False)

CSV_HEADER = CSV_HEADER + [
    "Title",
    "PeopleInTicket",
    "FarePerPerson",
]

COLUMN_DEFAULTS = build_column_defaults(
    titanic.NUMERICAL_CAT_CONT_KEYS + ["FarePerPerson"],
    titanic.NUMERICAL_CAT_NONCONT_KEYS + ["PeopleInTicket"],
    titanic.TARGET_FEATURE_NAME,
    csv_headers=CSV_HEADER,
)

TEXT_FEATURES_WITH_VOCABULARY = dict(
    map(
        lambda feature: (feature, list(train_csv[feature].unique())),
        titanic.TEXT_FEAT_KEYS + ["Title"],
    )
)
NUMERIC_FEATURES_WITH_VOCAB = dict(
    noncontinous=dict(
        map(
            lambda feature: (feature, list(train_csv[feature].unique())),
            titanic.NUMERICAL_CAT_NONCONT_KEYS + ["PeopleInTicket"],
        )
    ),
    continous=dict(
        map(
            lambda feature: (feature, train_csv[feature]),
            titanic.NUMERICAL_CAT_CONT_KEYS + ["FarePerPerson"],
        )
    ),
)

CATEGORICAL_FEATURES_WITH_VOCABULARY = dict(
    map(
        lambda feature: (feature, list(train_csv[feature].unique())),
        titanic.CATEGORICAL_FEAT_KEYS,
    )
)

CSV_FEATURES = {
    "NUMERIC_FEATURES_WITH_VOCABULARY": NUMERIC_FEATURES_WITH_VOCAB,
    "CATEGORICAL_FEATURES_WITH_VOCABULARY": CATEGORICAL_FEATURES_WITH_VOCABULARY,
    "TEXT_FEATURE_NAMES_WITH_VOCABULARY": TEXT_FEATURES_WITH_VOCABULARY,
    "TARGET_FEATURE_NAME": titanic.TARGET_FEATURE_NAME,
}

train_dataset = fetch_dataset_from_csv(
    csv_filepath=TRANSFORMED_TRAIN_CSV_PATH,
    batch_size=batch_size,
    csv_headers=CSV_HEADER,
    col_defaults=COLUMN_DEFAULTS,
    label_names=titanic.TARGET_FEATURE_NAME,
    shuffle=True,
)
test_dataset = fetch_dataset_from_csv(
    csv_filepath=TRANSFORMED_TEST_CSV_PATH, batch_size=batch_size
)

baseline_model = create_bl_model(
    hidden_units=hidden_units,
    dropout=dropout_rate,
    csv_feature_names=CSV_FEATURES,
    csv_data=train_csv,
)

keras.utils.plot_model(baseline_model, show_shapes=True, rankdir="LR")

baseline_model.compile(
    optimizer=optimizers.Adam(learning_rate=learning_rate),
    loss=losses.BinaryCrossentropy(from_logits=True),
    metrics=["accuracy"],
)

baseline_model.fit(
    train_dataset, epochs=50, verbose=2, workers=16, use_multiprocessing=True
)

import csv

predictions = baseline_model.predict(test_dataset, verbose=0)
# Round the predictions
rounded_predictions = [round(x[0]) for x in predictions]

# Write the predictions to a csv file
with open(SUBMISSION_PATH, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["PassengerId", "Survived"])
    for i, prediction in zip(test_csv["PassengerId"], rounded_predictions):
        writer.writerow([i, int(prediction)])
