"""
Probability Calibration Module
Fits Sigmoid / Platt scaling on validation decision scores only,
computes calibration curves, probability bins, Brier scores, and log loss.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from typing import Dict, Any, List, Optional, Tuple

SEED = 42


def fit_sigmoid_calibrator(
    val_scores: np.ndarray,
    y_val: np.ndarray,
    seed: int = 42
) -> LogisticRegression:
    """
    Fit a 1D logistic regression (Platt scaling) on VALIDATION decision scores.
    """
    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        random_state=seed
    )
    val_scores_2d = np.asarray(val_scores).reshape(-1, 1)
    y_val = np.asarray(y_val).astype(int)
    calibrator.fit(val_scores_2d, y_val)
    return calibrator


def predict_calibrated_probabilities(
    calibrator: LogisticRegression,
    scores: np.ndarray
) -> np.ndarray:
    """
    Convert raw decision scores into calibrated risk probabilities in [0, 1].
    """
    scores_2d = np.asarray(scores).reshape(-1, 1)
    return calibrator.predict_proba(scores_2d)[:, 1]


def evaluate_calibration(
    y_true: np.ndarray,
    calibrated_probs: np.ndarray,
    n_bins: int = 10
) -> Tuple[Dict[str, Any], pd.DataFrame, Dict[str, Any]]:
    """
    Compute calibration metrics, probability bin table, and reliability curve coordinates.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(calibrated_probs).ravel()

    # Metrics
    brier = float(brier_score_loss(y_true, probs))
    ll = float(log_loss(y_true, probs))
    auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else 0.5
    observed_prev = float(np.mean(y_true))
    mean_pred_prob = float(np.mean(probs))

    metrics = {
        "brier_score": brier,
        "log_loss": ll,
        "calibrated_roc_auc": auc,
        "observed_prevalence": observed_prev,
        "mean_predicted_probability": mean_pred_prob,
        "calibration_gap": abs(observed_prev - mean_pred_prob),
    }

    # Probability bins
    prob_df = pd.DataFrame({"probability": probs, "actual": y_true})
    bins = np.linspace(0, 1, n_bins + 1)
    prob_df["bin"] = pd.cut(prob_df["probability"], bins=bins, include_lowest=True)

    bin_summary = (
        prob_df.groupby("bin", observed=False)
               .agg(
                   samples=("actual", "size"),
                   mean_predicted_probability=("probability", "mean"),
                   observed_apnea_rate=("actual", "mean")
               )
               .reset_index()
    )
    bin_summary["bin_str"] = bin_summary["bin"].astype(str)

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=n_bins, strategy="uniform")
    curve_data = {
        "prob_pred": prob_pred.tolist(),
        "prob_true": prob_true.tolist(),
    }

    return metrics, bin_summary, curve_data
