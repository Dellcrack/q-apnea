"""
Model Evaluation and Metric Calculation Module
Calculates standard and clinical machine learning evaluation metrics,
confusion matrices, ROC curves, and Precision-Recall curves.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)
from typing import Dict, Any, List, Optional, Tuple


def evaluate_predictions(
    y_true: np.ndarray,
    scores: np.ndarray,
    split_name: str = "test",
    threshold: float = 0.0,
    model_name: str = "Model",
    representation: str = "4D_Latent",
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics at a given decision threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).ravel()
    y_pred = (scores >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Safety for zero-division
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    sens = float(recall_score(y_true, y_pred, zero_division=0))
    spec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, scores))
    except Exception:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, scores))
    except Exception:
        pr_auc = 0.0

    return {
        "model": model_name,
        "representation": representation,
        "split": split_name,
        "threshold": float(threshold),
        "accuracy": acc,
        "precision": prec,
        "sensitivity": sens,
        "specificity": spec,
        "f1": f1,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "total_samples": int(len(y_true)),
        "apnea_prevalence": float(np.mean(y_true)),
    }


def compute_roc_pr_curves(
    y_true: np.ndarray,
    scores: np.ndarray
) -> Dict[str, Any]:
    """
    Compute ROC (FPR, TPR) and PR (Precision, Recall) curves data.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).ravel()

    fpr, tpr, roc_thresholds = roc_curve(y_true, scores)
    precision, recall, pr_thresholds = precision_recall_curve(y_true, scores)

    return {
        "roc": {
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "thresholds": roc_thresholds.tolist(),
            "auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5,
        },
        "pr": {
            "precision": precision.tolist(),
            "recall": recall.tolist(),
            "thresholds": pr_thresholds.tolist(),
            "auc": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        }
    }


def search_optimal_thresholds(
    y_val: np.ndarray,
    val_scores: np.ndarray,
    n_points: int = 201,
    model_name: str = "QSVM",
    representation: str = "4D_Latent"
) -> Tuple[float, float, pd.DataFrame]:
    """
    Find optimal decision thresholds on the VALIDATION set using:
    1. Youden's J statistic: max(Sensitivity + Specificity - 1.0)
    2. F1-Score: max(F1)
    
    Returns:
    - best_youden_threshold
    - best_f1_threshold
    - threshold_analysis_df
    """
    y_val = np.asarray(y_val).astype(int)
    scores = np.asarray(val_scores).ravel()

    val_min = float(np.min(scores))
    val_max = float(np.max(scores))
    threshold_grid = np.linspace(val_min, val_max, n_points)

    rows = []
    for th in threshold_grid:
        metrics = evaluate_predictions(
            y_val, scores, "validation",
            threshold=float(th),
            model_name=model_name,
            representation=representation
        )
        metrics["youden_j"] = metrics["sensitivity"] + metrics["specificity"] - 1.0
        rows.append(metrics)

    df_th = pd.DataFrame(rows)
    youden_idx = df_th["youden_j"].idxmax()
    f1_idx = df_th["f1"].idxmax()

    best_youden_th = float(df_th.loc[youden_idx, "threshold"])
    best_f1_th = float(df_th.loc[f1_idx, "threshold"])

    return best_youden_th, best_f1_th, df_th
