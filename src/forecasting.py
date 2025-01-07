import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import logging
import os
import time
import warnings
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from src.utils import reset_seeds
import tensorflow as tf

from src.evaluation_metrics import forecast_metrics


os.environ["TF_DETERMINISTIC_OPS"] = "1"
warnings.filterwarnings(action="ignore", message="Setting attributes")

class TimeSeriesForecaster:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.back_horizon = None 
        self.horizon = None
        self.n_in_features = None
        self.n_out_features = None
        self.history = None
        self.mean_smape = None
        self.mean_rmse = None
        self.Y_preds = None

    def train_model(self, model_name, dataset, back_horizon, horizon):
        # Store parameters
        self.model_name = model_name
        self.back_horizon = back_horizon
        self.horizon = horizon
        self.n_in_features = dataset.X_train.shape[2]
        self.n_out_features = 1

        ###############################################
        # ## 2.0 Forecasting model
        ###############################################
        # reset seeds for numpy, tensorflow, python random package and python environment seed
        self.model = self._build_model()

        # Define the early stopping criteria
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", min_delta=0.0001, patience=10, restore_best_weights=True
        )

        # Train the model
        self.history = self.model.fit(
            dataset.X_train,
            dataset.Y_train,
            epochs=200,
            batch_size=64,
            validation_data=(dataset.X_val, dataset.Y_val),
            callbacks=[early_stopping],
        )

        # Predict on the testing set (forecast)
        self.Y_preds = self.model.predict(dataset.X_test)
        self.mean_smape, self.mean_rmse = forecast_metrics(dataset, self.Y_preds)

        print(
            f"[[{self.model_name}]] model trained, with test sMAPE score {self.mean_smape:0.4f}; test RMSE score: {self.mean_rmse:0.4f}."
        )

    def _build_model(self):
        reset_seeds(42)
        
        if self.model_name in ["wavenet", "seq2seq"]:
            return self._build_tfts_model()
        elif self.model_name == "gru":
            model = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Input(shape=(self.back_horizon, self.n_in_features)),
                    # Shape [batch, time, features] => [batch, time, gru_units]
                    tf.keras.layers.GRU(100, activation="tanh", return_sequences=True),
                    tf.keras.layers.GRU(50, activation="tanh", return_sequences=False),
                    # Shape => [batch, time, features]
                    tf.keras.layers.Dense(self.horizon, activation="linear"),
                    tf.keras.layers.Reshape((self.horizon, self.n_out_features)),
                ]
            )

            # Definition of the objective function and the optimizer
            optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
            model.compile(optimizer=optimizer, loss="mae")
            return model
        else:
            print("Not implemented: model_name.")
            return None
        
    def _build_tfts_model(self):
        import tfts

        inputs = tf.keras.layers.Input([self.back_horizon, self.n_in_features])
        if self.model_name == "wavenet":
            backbone = tfts.AutoModel(
                self.model_name,
                predict_length=self.horizon,
                custom_model_params={
                    "filters": 256,
                    "skip_connect_circle": True,
                },
            )
        elif self.model_name == "seq2seq":
            backbone = tfts.AutoModel(
                "seq2seq",
                predict_length=self.horizon,
                custom_model_params={"rnn_size": 256, "dense_size": 256},
            )
        else:
            print("Not implemented: build_tfts_model.")
        outputs = backbone(inputs)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        model.compile(optimizer=optimizer, loss="mae")

        return model

    def predict(self, x):
        """Predict using the underlying model
        Parameters
        ----------
        x : array-like
            Input samples
        Returns
        -------
        array-like
            Model predictions
        """
        return self.model.predict(x)
