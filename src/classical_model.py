"""
Classical Machine Learning Models Module
Implements classical RBF-SVM baseline models, hyperparameter tuning on validation set,
and matched 1,000-sample balanced baseline for fair comparison with QSVM.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List, Optional, Tuple
from src.evaluation import evaluate_predictions

SEED = 42


def fit_classical_rbf_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    C: float = 1.0,
    gamma: Any = "scale",
    class_weight: Any = "balanced",
    representation: str = "10D_HRV",
    model_name: str = "RBF-SVM",
    seed: int = 42
) -> Dict[str, Any]:
    """
    Train a classical RBF-SVM and evaluate on both validation and test sets.
    """
    model = SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        probability=False,
        random_state=seed,
    )
    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    fit_time = time.perf_counter() - t0

    val_scores = model.decision_function(X_val)
    test_scores = model.decision_function(X_test)

    val_metrics = evaluate_predictions(
        y_val, val_scores, "validation",
        threshold=0.0,
        model_name=model_name,
        representation=representation
    )
    test_metrics = evaluate_predictions(
        y_test, test_scores, "test",
        threshold=0.0,
        model_name=model_name,
        representation=representation
    )

    return {
        "model": model,
        "fit_time_sec": fit_time,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_scores": val_scores,
        "test_scores": test_scores,
        "C": C,
        "gamma": gamma,
    }


def tune_ae_svm_on_validation(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    c_grid: List[float] = [0.1, 1.0, 10.0],
    gamma_grid: List[Any] = ["scale", 0.01, 0.1, 1.0],
    seed: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Perform grid search for C and gamma exclusively on the VALIDATION set.
    """
    tuning_rows = []
    best_auc = -1.0
    best_params = {"C": 1.0, "gamma": "scale"}

    for C in c_grid:
        for gamma in gamma_grid:
            m = SVC(
                kernel="rbf",
                C=C,
                gamma=gamma,
                class_weight="balanced",
                probability=False,
                random_state=seed,
            )
            m.fit(X_train, y_train)
            v_scores = m.decision_function(X_val)
            v_metrics = evaluate_predictions(
                y_val, v_scores, "validation",
                threshold=0.0,
                model_name="AE-RBF-SVM-Tuned",
                representation="4D_Autoencoder"
            )
            v_metrics["C"] = C
            v_metrics["gamma"] = gamma
            tuning_rows.append(v_metrics)

            if v_metrics["roc_auc"] > best_auc:
                best_auc = v_metrics["roc_auc"]
                best_params = {"C": C, "gamma": gamma}

    tuning_df = pd.DataFrame(tuning_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # Train final model with best validation params
    final_res = fit_classical_rbf_svm(
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
        C=best_params["C"],
        gamma=best_params["gamma"],
        representation="4D_Autoencoder",
        model_name="AE-RBF-SVM-Tuned",
        seed=seed
    )

    return final_res, tuning_df


def fit_matched_classical_baseline(
    X_train_ae_unscaled: np.ndarray,
    y_train: np.ndarray,
    train_records: np.ndarray,
    X_val_ae_unscaled: np.ndarray,
    y_val: np.ndarray,
    X_test_ae_unscaled: np.ndarray,
    y_test: np.ndarray,
    qs_train_idx: np.ndarray,
    c_grid: List[float] = [0.1, 1.0, 10.0, 100.0],
    seed: int = 42
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """
    Matched 1,000-sample balanced classical baseline for direct comparison with QSVM.
    """
    matched_scaler = StandardScaler().fit(X_train_ae_unscaled[qs_train_idx])
    X_matched_train = matched_scaler.transform(X_train_ae_unscaled[qs_train_idx]).astype(np.float32)
    X_matched_val = matched_scaler.transform(X_val_ae_unscaled).astype(np.float32)
    X_matched_test = matched_scaler.transform(X_test_ae_unscaled).astype(np.float32)
    y_matched_train = y_train[qs_train_idx]

    tuning_rows = []
    best_auc = -1.0
    best_C = 1.0

    for C in c_grid:
        m = SVC(
            kernel="rbf",
            C=C,
            gamma="scale",
            probability=False,
            random_state=seed
        )
        m.fit(X_matched_train, y_matched_train)
        v_scores = m.decision_function(X_matched_val)
        row = evaluate_predictions(
            y_val, v_scores, "validation",
            model_name="Matched-AE-RBF-SVM",
            representation="4D_AE_1000_balanced"
        )
        row["C"] = C
        tuning_rows.append(row)
        if row["roc_auc"] > best_auc:
            best_auc = row["roc_auc"]
            best_C = C

    tuning_df = pd.DataFrame(tuning_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    final_model = SVC(
        kernel="rbf",
        C=best_C,
        gamma="scale",
        probability=False,
        random_state=seed
    )
    t0 = time.perf_counter()
    final_model.fit(X_matched_train, y_matched_train)
    fit_time = time.perf_counter() - t0

    val_scores = final_model.decision_function(X_matched_val)
    test_scores = final_model.decision_function(X_matched_test)

    val_metrics = evaluate_predictions(
        y_val, val_scores, "validation",
        model_name="Matched-AE-RBF-SVM",
        representation="4D_AE_1000_balanced"
    )
    test_metrics = evaluate_predictions(
        y_test, test_scores, "test",
        model_name="Matched-AE-RBF-SVM",
        representation="4D_AE_1000_balanced"
    )

    res = {
        "model": final_model,
        "scaler": matched_scaler,
        "fit_time_sec": fit_time,
        "best_C": best_C,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_scores": val_scores,
        "test_scores": test_scores,
    }
    return res, tuning_df
