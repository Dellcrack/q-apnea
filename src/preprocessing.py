"""
Preprocessing and Feature Selection Module
Implements record-level train/validation/test splitting, train-only median imputation,
standardization, correlation analysis, and feature selection.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from typing import Tuple, Dict, Any, List, Optional
from src.features import ALL_RAW_FEATURES, SELECTED_FEATURES


def record_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split dataset into Train, Validation, and Test sets by record groups.
    Ensures:
    1. Zero record leakage across splits.
    2. 'c05' and 'c06' are kept together in the same split (PhysioNet identical patient records).
    3. Stratification by apnea burden ratio.
    """
    if "record" not in df.columns:
        # If no record column, create deterministic record blocks
        df = df.copy()
        n_pseudo = min(35, max(3, len(df) // 100))
        df["record"] = [f"rec_{i % n_pseudo:02d}" for i in range(len(df))]

    record_stats = (
        df.groupby("record")
          .agg(
              total_minutes=("label", "size"),
              apnea_minutes=("label", "sum")
          )
          .reset_index()
    )

    record_stats["apnea_ratio"] = (
        record_stats["apnea_minutes"] / record_stats["total_minutes"]
    )

    record_stats["group"] = record_stats["record"]
    # Group c05 and c06 together if present
    record_stats.loc[
        record_stats["record"].isin(["c05", "c06"]),
        "group"
    ] = "c05_c06"

    group_stats = (
        record_stats.groupby("group")
                    .agg(
                        total_minutes=("total_minutes", "sum"),
                        apnea_minutes=("apnea_minutes", "sum")
                    )
                    .reset_index()
    )

    group_stats["apnea_ratio"] = (
        group_stats["apnea_minutes"] / group_stats["total_minutes"]
    )

    groups = group_stats["group"].to_numpy()

    # Stratify by apnea burden
    try:
        group_stats["stratification_bin"] = pd.qcut(
            group_stats["apnea_ratio"],
            q=min(4, max(2, len(group_stats) // 4)),
            labels=False,
            duplicates="drop"
        )
        train_groups, temp_groups = train_test_split(
            groups,
            test_size=1.0 - train_ratio,
            random_state=seed,
            stratify=group_stats["stratification_bin"]
        )

        temp_stats = group_stats[group_stats["group"].isin(temp_groups)].copy()
        test_prop = test_ratio / (val_ratio + test_ratio)
        val_groups, test_groups = train_test_split(
            temp_groups,
            test_size=test_prop,
            random_state=seed,
            stratify=temp_stats["stratification_bin"]
        )
    except Exception:
        # Deterministic fallback
        rng = np.random.default_rng(seed)
        shuffled = groups.copy()
        rng.shuffle(shuffled)
        n_train = max(1, int(round(train_ratio * len(shuffled))))
        train_groups = shuffled[:n_train]
        temp_groups = shuffled[n_train:]
        midpoint = max(1, len(temp_groups) // 2) if len(temp_groups) > 1 else len(temp_groups)
        val_groups = temp_groups[:midpoint]
        test_groups = temp_groups[midpoint:] if len(temp_groups) > 1 else temp_groups

    def expand_groups(group_list):
        expanded = []
        for group in group_list:
            if group == "c05_c06":
                expanded.extend(["c05", "c06"])
            else:
                expanded.append(group)
        return sorted(list(set(expanded)))

    train_records = expand_groups(train_groups)
    val_records = expand_groups(val_groups)
    test_records = expand_groups(test_groups)

    train_df = df[df["record"].isin(train_records)].copy().reset_index(drop=True)
    val_df = df[df["record"].isin(val_records)].copy().reset_index(drop=True)
    test_df = df[df["record"].isin(test_records)].copy().reset_index(drop=True)

    split_metadata = {
        "train_records": train_records,
        "val_records": val_records,
        "test_records": test_records,
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "test_samples": len(test_df),
        "train_apnea_ratio": float(train_df["label"].mean()) if len(train_df) else 0.0,
        "val_apnea_ratio": float(val_df["label"].mean()) if len(val_df) else 0.0,
        "test_apnea_ratio": float(test_df["label"].mean()) if len(test_df) else 0.0,
    }

    return train_df, val_df, test_df, split_metadata


def preprocess_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_subset: Optional[List[str]] = None
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray,
    StandardScaler, pd.Series, Dict[str, Any]
]:
    """
    Apply train-only median imputation and standardization.
    
    Returns:
    - X_train, X_val, X_test (standardized selected features)
    - y_train, y_val, y_test
    - fitted StandardScaler
    - train_medians Series
    - summary metrics dict
    """
    if feature_subset is None:
        feature_subset = SELECTED_FEATURES

    train_numeric = (
        train_df[feature_subset]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    val_numeric = (
        val_df[feature_subset]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    test_numeric = (
        test_df[feature_subset]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )

    # TRAIN ONLY medians
    train_medians = train_numeric.median()
    # Handle any potential remaining NaNs in train_medians with 0.0
    train_medians = train_medians.fillna(0.0)

    train_numeric = train_numeric.fillna(train_medians)
    val_numeric = val_numeric.fillna(train_medians)
    test_numeric = test_numeric.fillna(train_medians)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_numeric).astype(np.float32)
    X_val = scaler.transform(val_numeric).astype(np.float32)
    X_test = scaler.transform(test_numeric).astype(np.float32)

    y_train = train_df["label"].to_numpy(dtype=np.int64)
    y_val = val_df["label"].to_numpy(dtype=np.int64)
    y_test = test_df["label"].to_numpy(dtype=np.int64)

    summary = {
        "features": feature_subset,
        "n_features": len(feature_subset),
        "train_shape": X_train.shape,
        "val_shape": X_val.shape,
        "test_shape": X_test.shape,
        "means": scaler.mean_.tolist(),
        "scales": scaler.scale_.tolist(),
    }

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler, train_medians, summary


def compute_correlation_analysis(train_df: pd.DataFrame, features: List[str]) -> Dict[str, pd.DataFrame]:
    """
    Compute train-only correlation matrix, high-correlation pairs (>0.90),
    feature-target correlation, and mean redundancy.
    """
    numeric_df = (
        train_df[features]
        .apply(pd.to_numeric, errors="coerce")
        .replace([np.inf, -np.inf], np.nan)
    )
    medians = numeric_df.median().fillna(0.0)
    numeric_df = numeric_df.fillna(medians)

    corr_matrix = numeric_df.corr(method="pearson")

    # Highly correlated pairs
    high_pairs = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            c = corr_matrix.iloc[i, j]
            if abs(c) >= 0.90:
                high_pairs.append({
                    "Feature 1": features[i],
                    "Feature 2": features[j],
                    "Correlation": float(c),
                    "Absolute Correlation": float(abs(c)),
                })
    high_corr_df = pd.DataFrame(high_pairs)
    if not high_corr_df.empty:
        high_corr_df = high_corr_df.sort_values("Absolute Correlation", ascending=False).reset_index(drop=True)

    # Feature-target correlation
    target = train_df["label"].astype(float).to_numpy()
    target_corrs = []
    for feat in features:
        x = numeric_df[feat].to_numpy()
        c = np.corrcoef(x, target)[0, 1] if np.std(x) > 0 else 0.0
        target_corrs.append({
            "Feature": feat,
            "Target Correlation": float(c),
            "Absolute Target Correlation": float(abs(c)),
        })
    target_corr_df = pd.DataFrame(target_corrs).sort_values("Absolute Target Correlation", ascending=False).reset_index(drop=True)

    # Redundancy (mean correlation with other features)
    mean_abs = corr_matrix.abs().where(~np.eye(len(features), dtype=bool)).mean(axis=1)
    redundancy_df = pd.DataFrame({
        "Feature": features,
        "Mean Absolute Correlation": mean_abs.values
    }).sort_values("Mean Absolute Correlation", ascending=False).reset_index(drop=True)

    return {
        "corr_matrix": corr_matrix,
        "high_corr_df": high_corr_df,
        "target_corr_df": target_corr_df,
        "redundancy_df": redundancy_df,
    }


def fit_pca_4d(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, PCA, pd.DataFrame]:
    """
    Fit 4D PCA on training data only and transform validation and test sets.
    """
    pca = PCA(n_components=4, random_state=seed)
    X_tr_pca = pca.fit_transform(X_train).astype(np.float32)
    X_v_pca = pca.transform(X_val).astype(np.float32)
    X_te_pca = pca.transform(X_test).astype(np.float32)

    variance_df = pd.DataFrame({
        "Component": [f"PC{i+1}" for i in range(4)],
        "Explained Variance Ratio": pca.explained_variance_ratio_,
        "Cumulative Variance": np.cumsum(pca.explained_variance_ratio_),
    })

    return X_tr_pca, X_v_pca, X_te_pca, pca, variance_df
