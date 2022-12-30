import datetime

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import BatchNormalization, Dense
from keras.models import Sequential
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from common.preprocessor import (
    load_playground_data,
    get_missings,
    rmse,
    high_correlated,
)

if __name__ == "__main__":
    data, submissions = load_playground_data()

    features_with_Nan = get_missings(data).Column.to_list()
    print(f"There are {len(features_with_Nan)} features with Nan values")

    loss_per_feature = {}
    data_imputed = pd.DataFrame()

    for col in tqdm(data[features_with_Nan].columns):
        predictions = []
        validation_loss = []

        not_null = ~data[col].isnull()
        train = data.loc[not_null]
        test = data.loc[~not_null]

        selected_features = [
            n for n in high_correlated(col, data) if n not in ["row_id", col]
        ]
        kf = KFold(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(kf.split(train[selected_features])):
            tf.keras.backend.clear_session()
            tf.random.set_seed(42)

            X_train, X_val = train.iloc[train_idx].drop(col, axis=1), train.iloc[
                val_idx
            ].drop(col, axis=1)
            y_train, y_val = train.iloc[train_idx][col], train.iloc[val_idx][col]

            X_test = test.drop(col, axis=1)
            X_train, X_val = X_train.fillna(X_train.median()), X_val.fillna(
                X_val.median()
            )
            X_test = X_test.fillna(X_test.median())

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_val = scaler.transform(X_val)
            X_test = scaler.transform(X_test)

            # Create a sequential model
            model = Sequential(
                [
                    tf.keras.layers.Input(shape=X_train.shape[1:]),
                    Dense(512, activation="swish"),
                    BatchNormalization(),
                    Dense(256, activation="swish"),
                    BatchNormalization(),
                    Dense(128, activation="swish"),
                    BatchNormalization(),
                    Dense(64, activation="swish"),
                    BatchNormalization(),
                    Dense(32, activation="swish"),
                    BatchNormalization(),
                    Dense(16, activation="swish"),
                    BatchNormalization(),
                    Dense(1, activation="linear"),
                ]
            )
            # Compile the model
            model.compile(loss=rmse, optimizer=Adam(learning_rate=0.01), metrics=[rmse])
            # Define callbacks
            lr = ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=3, verbose=0
            )
            es = EarlyStopping(
                monitor="val_loss",
                patience=12,
                verbose=0,
                mode="min",
                restore_best_weights=True,
            )
            log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
            callbacks = [lr, es]

            # Fit the model
            history = model.fit(
                X_train,
                y_train,
                epochs=30,
                validation_data=(X_val, y_val),
                batch_size=4094,
                shuffle=True,
                callbacks=callbacks,
                verbose=0,
                workers=18,
                use_multiprocessing=True,
            )

            y_preds = model.predict(X_test)
            predictions.append(y_preds)
            validation_loss.append(history.history["val_loss"][-1])

        mean_values = np.array(predictions).mean(axis=0)
        loss_per_feature[col] = np.mean(validation_loss)
        # Specifying column to impute
        imputed_feature = data[col].copy()

        # Filling missing values
        imputed_feature.loc[~not_null] = mean_values.ravel()

        # Concatenate imputed columns
        data_imputed = pd.concat([data_imputed, imputed_feature], axis=1)
    data[features_with_Nan] = data_imputed

    loss_df = pd.DataFrame(loss_per_feature, index=["Validation_RMSE"]).T.sort_values(
        by="Validation_RMSE", ascending=False
    )

    print(loss_df)

    for i in submissions.index:
        row = int(i.split("-")[0])
        col = i.split("-")[1]
        submissions.loc[i, "value"] = data.loc[row, col]

    submissions.to_csv("submission.csv")
