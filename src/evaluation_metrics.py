import numpy as np
import pandas as pd

def forecast_metrics(dataset, Y_pred, inverse_transform=True):
    Y_test_original, Y_pred_original = list(), list()
    if inverse_transform:
        for i in range(dataset.X_test.shape[0]):
            idx = dataset.test_idxs[i]
            scaler = dataset.scaler[idx]

            Y_test_original.append(
                scaler[dataset.target_col].inverse_transform(dataset.Y_test[i])
            )
            Y_pred_original.append(
                scaler[dataset.target_col].inverse_transform(Y_pred[i])
            )

        Y_test_original = np.array(Y_test_original)
        Y_pred_original = np.array(Y_pred_original)
    else:
        Y_test_original = dataset.Y_test
        Y_pred_original = Y_pred

    def smape(Y_test, Y_pred):
        # src: https://github.com/ServiceNow/N-BEATS/blob/c746a4f13ffc957487e0c3279b182c3030836053/common/metrics.py
        def smape_sample(actual, forecast):
            return 200 * np.mean(
                np.abs(forecast - actual) / (np.abs(actual) + np.abs(forecast))
            )

        return np.mean([smape_sample(Y_test[i], Y_pred[i]) for i in range(len(Y_pred))])

    def rmse(Y_test, Y_pred):
        return np.sqrt(np.mean((Y_pred - Y_test) ** 2))

    mean_smape = smape(Y_test_original, Y_pred_original)
    mean_rmse = rmse(Y_test_original, Y_pred_original)

    return mean_smape, mean_rmse

def cf_metrics(
    desired_max_lst,
    desired_min_lst,
    X_test,
    cf_samples,
    z_preds,
    change_idx,
    hist_inputs,
):
    validity = validity_ratio(
        pred_values=z_preds,
        desired_max_lst=desired_max_lst,
        desired_min_lst=desired_min_lst,
    )
    proximity = euclidean_distance(
        X=X_test, cf_samples=cf_samples, change_idx=change_idx
    )
    compactness = compactness_score(
        X=X_test, cf_samples=cf_samples, change_idx=change_idx
    )
    cumsum_auc, cumsum_valid_steps, cumsum_counts = cumulative_valid_steps(
        pred_values=z_preds, max_bounds=desired_max_lst, min_bounds=desired_min_lst
    )

    # Auxiliary metrics for validating the historical values mechanism
    proximity_hist = mae_distance_hist(
        hist_inputs=hist_inputs, cf_samples=cf_samples, change_idx=change_idx
    )
    compactness_hist = compact_score_hist(
        hist_inputs=hist_inputs, cf_samples=cf_samples, change_idx=change_idx
    )

    return (
        validity,
        proximity,
        compactness,
        cumsum_valid_steps,
        cumsum_counts,
        cumsum_auc,
        proximity_hist,
        compactness_hist,
    )


# validity ratio
def validity_ratio(pred_values, desired_max_lst, desired_min_lst):
    validity_lst = np.logical_and(
        pred_values <= desired_max_lst, pred_values >= desired_min_lst
    ).mean(axis=1)
    return validity_lst.mean()


def cumulative_valid_steps(pred_values, max_bounds, min_bounds):
    input_array = np.logical_and(pred_values <= max_bounds, pred_values >= min_bounds)
    until_steps_valid = np.empty(input_array.shape[0])
    n_samples, n_steps_total, _ = pred_values.shape
    for i in range(input_array.shape[0]):
        step_counts = 0
        for step in range(input_array.shape[1]):
            if input_array[i, step] == True:
                step_counts += 1
                until_steps_valid[i] = step_counts
            elif input_array[i, step] == False:
                until_steps_valid[i] = step_counts
                break
            else:
                print("Wrong input: cumulative_valid_steps.")

    valid_steps, counts = np.unique(until_steps_valid, return_counts=True)
    cumsum_counts = np.flip(np.cumsum(np.flip(counts)))
    # remove the valid_step=0 (no valid cf preds) in the trapz calculation
    valid_steps, cumsum_counts = fillna_cumsum_counts(
        n_steps_total, valid_steps, cumsum_counts
    )

    cumsum_auc = np.trapz(
        cumsum_counts[1:] / n_samples, valid_steps[1:] / n_steps_total
    )

    return cumsum_auc, valid_steps, cumsum_counts


def fillna_cumsum_counts(n_steps_total, valid_steps, cumsum_counts):
    df = pd.DataFrame(
        [{key: val for key, val in zip(valid_steps, cumsum_counts)}],
        columns=list(range(0, n_steps_total + 1)),
    )
    df = df.sort_index(ascending=True, axis=1)
    # backfill the previous valid steps
    df = df.fillna(method="backfill", axis=1)
    # fill 0s for the right hand nas
    df = df.fillna(method=None, value=0)
    valid_steps, cumsum_counts = df.columns.to_numpy(), df.values[0]
    return valid_steps, cumsum_counts


def euclidean_distance(X, cf_samples, change_idx, average=True):
    X = X[:, :, change_idx]
    cf_samples = cf_samples[:, :, change_idx]
    paired_distances = np.linalg.norm(X - cf_samples, axis=1)
    # return the average of compactness for each sample
    return paired_distances.mean(axis=1).mean() if average else paired_distances


# originally from: https://github.com/isaksamsten/wildboar/blob/859758884677ba32a601c53a5e2b9203a644aa9c/src/wildboar/metrics/_counterfactual.py#L279
def compactness_score(X, cf_samples, change_idx):
    X = X[:, :, change_idx]
    cf_samples = cf_samples[:, :, change_idx]
    # absolute tolerance atol=0.01, 0.001, OR 0.0001?
    c = np.isclose(X, cf_samples, atol=0.001)
    compact_lst = np.mean(c, axis=1)
    # return the average of compactness for each sample
    return compact_lst.mean(axis=1).mean()


def mae_distance_hist(hist_inputs, cf_samples, change_idx):
    mae_scores = list()
    for i in range(cf_samples.shape[0]):  # for each sample, i
        hist_sample = hist_inputs[i]
        min_ae_lst = list()
        for j in change_idx:  # for each feature, j
            cf_repeated = np.repeat(
                cf_samples[i, :, j][:, np.newaxis], len(hist_sample[j]), axis=1
            )
            min_dist = np.nanmin(np.abs(cf_repeated - hist_sample[j]), axis=1)
            min_ae_lst.append(min_dist.mean())
        mae_per_sample = np.mean(min_ae_lst)
        mae_scores.append(mae_per_sample)
    return np.mean(mae_scores)


def compact_score_hist(hist_inputs, cf_samples, change_idx):
    compact_scores = list()
    for i in range(cf_samples.shape[0]):  # for each sample, i
        hist_sample = hist_inputs[i]
        c_lst = list()
        for j in change_idx:  # for each feature, j
            cf_repeated = np.repeat(
                cf_samples[i, :, j][:, np.newaxis], len(hist_sample[j]), axis=1
            )
            # absolute tolerance atol; defined for hist_inputs (decimals=2)
            atol = 0.005
            c_per_step = np.any(
                np.abs(cf_repeated - hist_sample[j]) <= atol, axis=1
            ).astype(np.float32)
            c_lst.append(c_per_step.mean())
        compact_per_sample = np.mean(c_lst)
        compact_scores.append(compact_per_sample)
    return np.mean(compact_scores)