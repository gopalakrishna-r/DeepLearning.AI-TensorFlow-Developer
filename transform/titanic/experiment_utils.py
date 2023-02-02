import itertools

import keras
from keras import layers
from keras import optimizers, losses, metrics

from preprocessor import create_model_inputs, encode_inputs, encode_inputs_with_lookups


def run_experiment(model, train_dataset, test_dataset):
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss=losses.SparseCategoricalCrossentropy(),
        metrics=[metrics.SparseCategoricalAccuracy()],
    )

    model.fit(train_dataset, epochs=50, verbose=2, workers=16, use_multiprocessing=True)

    _, accuracy = model.evaluate(test_dataset, verbose=1)

    print(f"Test accuracy {round(accuracy * 100, 2)}%")


def create_baseline_model(hidden_units, dropout, num_classes, csv_feature_names):
    numeric_feature_names_ = csv_feature_names["NUMERIC_FEATURE_NAMES"]
    categorical_feature_vocab_ = csv_feature_names[
        "CATEGORICAL_FEATURES_WITH_VOCABULARY"
    ]
    categorical_feature_names_ = list(categorical_feature_vocab_.keys())

    inputs = create_model_inputs(categorical_feature_names_, [], numeric_feature_names_)
    features = encode_inputs(
        inputs,
        categorical_feature_names_,
        categorical_feature_vocab_,
        use_embedding=True,
    )
    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout)(features)

    outputs = layers.Dense(units=num_classes, activation="sigmoid")(features)
    return keras.Model(inputs=inputs, outputs=outputs)


def create_bl_model(hidden_units, dropout, csv_feature_names, csv_data):
    numeric_feature_vocab_ = csv_feature_names["NUMERIC_FEATURES_WITH_VOCABULARY"]
    categorical_feature_vocab_ = csv_feature_names[
        "CATEGORICAL_FEATURES_WITH_VOCABULARY"
    ]
    textual_feature_vocab_ = csv_feature_names["TEXT_FEATURE_NAMES_WITH_VOCABULARY"]

    inputs = create_model_inputs(
        list(categorical_feature_vocab_.keys()),
        list(textual_feature_vocab_.keys()),
        list(
            itertools.chain.from_iterable(
                list(
                    map(
                        lambda feature_dict: feature_dict.keys(),
                        numeric_feature_vocab_.values(),
                    )
                )
            )
        ),
    )
    features = encode_inputs_with_lookups(inputs, csv_feature_names, csv_data)
    for units in hidden_units:
        features = layers.Dense(units)(features)
        features = layers.BatchNormalization()(features)
        features = layers.ReLU()(features)
        features = layers.Dropout(dropout)(features)

    outputs = layers.Dense(units=1, activation="sigmoid")(features)
    return keras.Model(inputs=inputs, outputs=outputs)


def build_column_defaults(
    continous_features, non_continous_features, target_feature_name, csv_headers
):
    COLUMN_DEFAULTS = []
    for feature_name in csv_headers:
        if feature_name in non_continous_features + [target_feature_name]:
            COLUMN_DEFAULTS.append([0])
        elif feature_name in continous_features:
            COLUMN_DEFAULTS.append([0.0])
        else:
            COLUMN_DEFAULTS.append(["UNK"])

    return COLUMN_DEFAULTS
