"""Evaluation metrics for match outcome prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score


def log_loss_score(y_true: pd.Series | list, y_pred_probs: pd.DataFrame | np.ndarray) -> float:
    """Compute multi-class log loss.

    Parameters
    ----------
    y_true : pd.Series or list
        True class labels ('home_win', 'draw', 'away_win').
    y_pred_probs : pd.DataFrame or np.ndarray
        Predicted probabilities for each class.
        If DataFrame, columns should be ['home_win_prob', 'draw_prob', 'away_win_prob'].

    Returns
    -------
    float
        Log loss value (lower is better).
    """
    if isinstance(y_true, pd.Series):
        y_true_list = y_true.tolist()
    else:
        y_true_list = list(y_true)

    if isinstance(y_pred_probs, pd.DataFrame):
        probs_array = y_pred_probs[["away_win_prob", "draw_prob", "home_win_prob"]].values
    else:
        probs_array = np.asarray(y_pred_probs)

    labels = ["away_win", "draw", "home_win"]
    return float(log_loss(y_true_list, probs_array, labels=labels))


def brier_score(y_true: pd.Series | list, y_pred_probs: pd.DataFrame | np.ndarray) -> float:
    """Compute multi-class Brier score.

    The Brier score measures the accuracy of probabilistic predictions.
    For multi-class with one-hot encoding:
        BS = (1/N) * sum_i sum_k (p_ik - y_ik)^2

    Parameters
    ----------
    y_true : pd.Series or list
        True class labels ('home_win', 'draw', 'away_win').
    y_pred_probs : pd.DataFrame or np.ndarray
        Predicted probabilities for each class.
        If DataFrame, columns should be ['home_win_prob', 'draw_prob', 'away_win_prob'].

    Returns
    -------
    float
        Brier score (lower is better, range [0, 2] for 3 classes).
    """
    if isinstance(y_true, pd.Series):
        y_true_list = y_true.tolist()
    else:
        y_true_list = list(y_true)

    if isinstance(y_pred_probs, pd.DataFrame):
        probs_array = y_pred_probs[["home_win_prob", "draw_prob", "away_win_prob"]].values
    else:
        probs_array = np.asarray(y_pred_probs)

    classes = ["home_win", "draw", "away_win"]
    n_samples = len(y_true_list)
    n_classes = len(classes)

    y_true_onehot = np.zeros((n_samples, n_classes))
    for i, label in enumerate(y_true_list):
        if label in classes:
            j = classes.index(label)
            y_true_onehot[i, j] = 1.0

    squared_errors = (probs_array - y_true_onehot) ** 2
    return float(squared_errors.sum() / n_samples)


def accuracy(y_true: pd.Series | list, y_pred_labels: pd.Series | list) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    y_true : pd.Series or list
        True class labels.
    y_pred_labels : pd.Series or list
        Predicted class labels.

    Returns
    -------
    float
        Accuracy score (higher is better).
    """
    if isinstance(y_true, pd.Series):
        y_true_list = y_true.tolist()
    else:
        y_true_list = list(y_true)

    if isinstance(y_pred_labels, pd.Series):
        y_pred_list = y_pred_labels.tolist()
    else:
        y_pred_list = list(y_pred_labels)

    return float(accuracy_score(y_true_list, y_pred_list))


def predict_to_labels(probs_df: pd.DataFrame) -> pd.Series:
    """Convert probability predictions to class labels.

    Parameters
    ----------
    probs_df : pd.DataFrame
        DataFrame with columns ['home_win_prob', 'draw_prob', 'away_win_prob'].

    Returns
    -------
    pd.Series
        Predicted class labels.
    """
    class_names = ["home_win_prob", "draw_prob", "away_win_prob"]
    label_names = ["home_win", "draw", "away_win"]
    return probs_df[class_names].idxmax(axis=1).map(
        dict(zip(class_names, label_names))
    )
