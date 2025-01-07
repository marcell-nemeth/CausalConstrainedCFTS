import copy
import csv
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class DataLoader:
    """
    Load data into desired formats for training/validation/testing, including preprocessing.
    """

    def __init__(self, horizon, back_horizon):
        self.horizon = horizon
        self.back_horizon = back_horizon
        self.scaler = list()
        self.historical_values = list()

    def preprocessing(
        self,
        lst_train_arrays,
        lst_test_arrays,
        # train_mode=True, # flag for train_mode (split into train/val), test_mode (no split)
        train_size=0.8,
        normalize=False,
        sequence_stride=6,
        target_col=0,
    ):
        self.normalize = normalize
        self.sequence_stride = sequence_stride
        self.target_col = target_col
        train_arrays = copy.deepcopy(lst_train_arrays)
        test_arrays = copy.deepcopy(lst_test_arrays)

        # count valid timesteps for each individual series
        # train_array.shape = n_timesteps x n_features
        self.valid_steps_train = [train_array.shape[0] for train_array in train_arrays]
        train_lst, val_lst, test_lst = list(), list(), list()
        for idx in range(len(train_arrays)):
            bg_sample_train = train_arrays[idx]
            bg_sample_test = test_arrays[idx]
            valid_steps_sample = self.valid_steps_train[idx]

            train = bg_sample_train[: int(train_size * valid_steps_sample), :].copy()
            val = bg_sample_train[int(train_size * valid_steps_sample) :, :].copy()
            test = bg_sample_test[:, :].copy()

            if self.normalize:
                scaler_cols = list()
                # train.shape = n_train_timesteps x n_features
                for col_idx in range(train.shape[1]):
                    scaler = MinMaxScaler(feature_range=(0, 1), clip=False)
                    train[:, col_idx] = remove_extra_dim(
                        scaler.fit_transform((add_extra_dim(train[:, col_idx])))
                    )
                    val[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(val[:, col_idx]))
                    )
                    test[:, col_idx] = remove_extra_dim(
                        scaler.transform(add_extra_dim(test[:, col_idx]))
                    )
                    scaler_cols.append(scaler)  # by col_idx, each feature
                self.scaler.append(scaler_cols)  # by pat_idx, each patient

            lst_hist_values = list()
            for col_idx in range(train.shape[1]):
                all_train_col = np.concatenate((train[:, col_idx], val[:, col_idx]))
                # decimals = 1, 2 OR 3?
                unique_values = np.unique(np.round(all_train_col, decimals=2))
                lst_hist_values.append(unique_values)
            self.historical_values.append(lst_hist_values)

            train_lst.append(train)
            val_lst.append(val)
            test_lst.append(test)

        (
            self.X_train,
            self.Y_train,
            self.train_idxs,
        ) = self.create_sequences(
            train_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_val,
            self.Y_val,
            self.val_idxs,
        ) = self.create_sequences(
            val_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )
        (
            self.X_test,
            self.Y_test,
            self.test_idxs,
        ) = self.create_sequences(
            test_lst,
            self.horizon,
            self.back_horizon,
            self.sequence_stride,
            self.target_col,
        )

    @staticmethod
    def create_sequences(
        series_lst, horizon, back_horizon, sequence_stride, target_col=0
    ):
        Xs, Ys, sample_idxs = list(), list(), list()

        cnt_nans = 0
        for idx, series in enumerate(series_lst):
            len_series = series.shape[0]
            if len_series < (horizon + back_horizon):
                print(
                    f"Warning: not enough timesteps to split for sample {idx}, len: {len_series}, horizon: {horizon}, back: {back_horizon}."
                )

            for i in range(0, len_series - back_horizon - horizon, sequence_stride):
                # all columns except the target column
                input_series = series[i : (i + back_horizon), :target_col]
                output_series = series[
                    (i + back_horizon) : (i + back_horizon + horizon), [target_col]
                ]
                if np.isfinite(input_series).all() and np.isfinite(output_series).all():
                    Xs.append(input_series)
                    Ys.append(output_series)
                    # record the sample index when splitting
                    sample_idxs.append(idx)
                else:
                    cnt_nans += 1
                    if cnt_nans % 100 == 0:
                        print(f"{cnt_nans} strides skipped due to NaN values.")

        return np.array(Xs), np.array(Ys), np.array(sample_idxs)


def load_dataset(dataset, data_path):
    if dataset == "simulated":
        # load into list of training arrays, and a standalone list for test
        lst_arrays_train, lst_arrays_test = load_sim_data(data_path, "causal_4cols_1000rows.csv")
    else:
        print("Not implemented: load_dataset.")
    return lst_arrays_train, lst_arrays_test


def load_sim_data(data_path, file_name="causal_4cols_1000rows.csv", train_size=0.8):
    data = pd.read_csv(data_path + "synthetic/" + file_name)
    
    if 'Unnamed: 0' in data.columns:
        data = data.drop(columns=['Unnamed: 0'])

    lst_arrays_train = list()
    lst_arrays_test = list()

    # split time series into train and test
    data_train = data.iloc[:int(train_size * data.shape[0]), :]
    data_test = data.iloc[int(train_size * data.shape[0]):, :]

    lst_arrays_train.append(
        np.asarray(
            data_train
        )
    )

    lst_arrays_test.append(
        np.asarray(
            data_test
        )
    )
    return lst_arrays_train, lst_arrays_test

# remove an extra dimension
def remove_extra_dim(input_array):
    # 2d to 1d
    if len(input_array.shape) == 2:
        return np.reshape(input_array, (-1))
    # 3d to 2d (remove the last empty dim)
    elif len(input_array.shape) == 3:
        return np.squeeze(np.asarray(input_array), axis=-1)
    else:
        print("Not implemented.")


# add an extra dimension
def add_extra_dim(input_array):
    # 1d to 2d
    if len(input_array.shape) == 1:
        return np.reshape(input_array, (-1, 1))
    # 2d to 3d
    elif len(input_array.shape) == 2:
        return np.asarray(input_array)[:, :, np.newaxis]
    else:
        print("Not implemented.")