import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import time
import numpy as np
from src.evaluation_metrics import cf_metrics
import tensorflow as tf

from src.data_loader import add_extra_dim
from src.sample_transform import BGForecastCF

class CounterfactualGenerator:
    def __init__(self, dataset, forecast_model, target_col, change_cols, random_state, horizon):
        self.dataset = dataset
        self.forecast_model = forecast_model
        self.TARGET_COL = target_col
        self.CHANGE_COLS = change_cols
        self.RANDOM_STATE = random_state
        self.horizon = horizon
        self.lower_bound = 0
        self.upper_bound = 1


    def generate_counterfactuals(self, model_name):
        rand_test_idx = np.arange(self.dataset.X_test.shape[0])
        X_test = self.dataset.X_test[rand_test_idx]
        Y_test = self.dataset.Y_test[rand_test_idx]

        print(f"Generating CFs for {len(rand_test_idx)} samples in total...")

        desired_max_lst, desired_min_lst = list(), list()
        hist_inputs = list()

        # define the desired center to reach
        desired_steps = 10
        desired_center = (self.upper_bound + self.lower_bound) / 2
        print(f"Desired center {desired_center} in {desired_steps} timesteps.")

        for i in range(len(X_test)):
            idx = self.dataset.test_idxs[rand_test_idx[i]]
            scaler = self.dataset.scaler[idx]

            desired_center_scaled = scaler[self.TARGET_COL].transform(
                np.array(desired_center).reshape(-1, 1)
            )[0][0]
            print(f"Desired center: {desired_center}; after scaling: {desired_center_scaled:0.4f}")

            center = "last"
            desired_shift, poly_order = 0, 1
            fraction_std = 0.5

            desired_max_scaled, desired_min_scaled = self._generate_bounds(
                center=center,
                shift=desired_shift,
                desired_center=desired_center_scaled,
                poly_order=poly_order,
                horizon=self.horizon,
                fraction_std=fraction_std,
                input_series=Y_test[i, :, 0],
                desired_steps=desired_steps,
            )

            desired_max_lst.append(desired_max_scaled)
            desired_min_lst.append(desired_min_scaled)
            hist_inputs.append(self.dataset.historical_values[idx])

        cf_model = self._setup_cf_model(model_name)
        
        start_time = time.time()
        cf_samples, losses, _ = cf_model.transform(
            X_test,
            desired_max_lst,
            desired_min_lst,
            clip_range_inputs=None,
            hist_value_inputs=None,
        )
        end_time = time.time()
        elapsed_time1 = end_time - start_time
        print(f"Elapsed time - ForecastCF: {elapsed_time1:0.4f}.")

        self._evaluate_counterfactuals(cf_samples, desired_max_lst, desired_min_lst, X_test, hist_inputs)

    def _setup_cf_model(self, model_name):
        cf_model = BGForecastCF(
            max_iter=10,
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
            pred_margin_weight=0.9,
            step_weights="unconstrained",
            random_state=self.RANDOM_STATE,
            target_col=self.TARGET_COL,
            only_change_idx=self.CHANGE_COLS,
        )

        if model_name in ["wavenet", "seq2seq", "gru"]:
            cf_model.fit(self.forecast_model)
        else:
            print("Not implemented: cf_model.fit.")
            
        return cf_model

    def _evaluate_counterfactuals(self, cf_samples, desired_max_lst, desired_min_lst, X_test, hist_inputs):
        cf_samples_lst = [cf_samples]
        CF_MODEL_NAMES = ["ForecastCF"]

        for i in range(len(cf_samples_lst)):
            z_preds = self.forecast_model.predict(cf_samples_lst[i])

            (
                validity,
                proximity,
                compactness,
                cumsum_valid_steps,
                cumsum_counts,
                cumsum_auc,
                proximity_hist,
                compactness_hist,
            ) = cf_metrics(
                desired_max_lst=desired_max_lst,
                desired_min_lst=desired_min_lst,
                X_test=X_test,
                cf_samples=cf_samples_lst[i],
                z_preds=z_preds,
                change_idx=self.CHANGE_COLS,
                hist_inputs=hist_inputs,
            )

            print(f"Done for CF search: [[{CF_MODEL_NAMES[i]}]].")
            print(f"validity: {validity}, step_validity_auc: {cumsum_auc}.")
            print(f"valid_steps: {cumsum_valid_steps}, counts:{cumsum_counts}.")
            print(f"proximity: {proximity}, compactness: {compactness}.")
            print(f"proximity_hist:{proximity_hist}, compactness_hist:{compactness_hist}")

    def _generate_bounds(self, center, shift, desired_center, poly_order, horizon, fraction_std, input_series, desired_steps):
        if center == "last":
            start_value = input_series[-1]
        elif center == "median":
            start_value = np.median(input_series)
        elif center == "mean":
            start_value = np.mean(input_series)
        elif center == "min":
            start_value = np.min(input_series)
        elif center == "max":
            start_value = np.max(input_series)
        else:
            print("Center: not implemented.")

        std = np.std(input_series)

        # Calculate the change_percent based on the desired center (in 2 hours)
        change_percent = (desired_center - start_value) / start_value
        # Create a default fluctuating range for the upper and lower bound if std is too small
        fluct_range = fraction_std * std if fraction_std * std >= 0.025 else 0.025
        upper = add_extra_dim(
            start_value
            * (
                1
                + self._polynomial_values(
                    shift, change_percent, poly_order, horizon, desired_steps
                )
                + fluct_range
            )
        )
        lower = add_extra_dim(
            start_value
            * (
                1
                + self._polynomial_values(
                    shift, change_percent, poly_order, horizon, desired_steps
                )
                - fluct_range
            )
        )

        return upper, lower

    def _polynomial_values(self, shift, change_percent, poly_order, horizon, desired_steps=None):
        """
        shift: e.g., +0.1 (110% of the start value)
        change_percent: e.g., 0.1 (10% increase)
        poly_order: e.g., order 1, or 2, ...
        horizon: the forecasting horizon
        desired_steps: the desired timesteps for the change_percent to finally happen (can be larger than horizon)
        """
        if horizon == 1:
            return np.asarray([shift + change_percent])
            
        desired_steps = desired_steps if desired_steps else horizon

        p_orders = [shift]  # intercept
        p_orders.extend([0 for i in range(poly_order)])
        p_orders[-1] = change_percent / ((desired_steps - 1) ** poly_order)

        p = np.polynomial.Polynomial(p_orders)
        p_coefs = list(reversed(p.coef))
        value_lst = np.asarray([np.polyval(p_coefs, i) for i in range(desired_steps)])

        return value_lst[:horizon]