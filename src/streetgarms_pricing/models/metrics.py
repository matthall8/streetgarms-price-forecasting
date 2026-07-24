"""Metrics for log-price predictions. Inputs (y_true, y_pred) are in LOG space.

RMSE in log space == RMSLE on price (a percentage-style error). MAPE / median-APE
are reported on the original £ scale for interpretability.
"""
import numpy as np


def rmse_log(y_true_log, y_pred_log) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_pred_log) - np.asarray(y_true_log)) ** 2)))


def _ape(y_true_log, y_pred_log):
    yt, yp = np.exp(np.asarray(y_true_log)), np.exp(np.asarray(y_pred_log))
    return np.abs(yp - yt) / yt


def mape(y_true_log, y_pred_log) -> float:
    return float(np.mean(_ape(y_true_log, y_pred_log)))


def median_ape(y_true_log, y_pred_log) -> float:
    return float(np.median(_ape(y_true_log, y_pred_log)))


def report(y_true_log, y_pred_log) -> dict:
    return {
        "rmse_log": rmse_log(y_true_log, y_pred_log),
        "mape": mape(y_true_log, y_pred_log),
        "median_ape": median_ape(y_true_log, y_pred_log),
    }
