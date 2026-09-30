"""
Quantum Support Vector Machine (QSVM) Module
Implements Qiskit ZZFeatureMap quantum encoding, quantum angle scaling to [0, pi],
Statevector simulation, Fidelity Quantum Kernel evaluation, and precomputed SVC training.

Simulator-based implementation (Qiskit Statevector simulation).
Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import zz_feature_map
from qiskit.quantum_info import Statevector
from typing import Dict, Any, List, Optional, Tuple, Callable
from src.evaluation import evaluate_predictions

N_QUBITS = 4
SEED = 42


def build_zz_feature_map(n_qubits: int = 4, reps: int = 2, entanglement: str = "linear"):
    """
    Construct Qiskit ZZFeatureMap for 4-qubit quantum state encoding.
    """
    return zz_feature_map(
        feature_dimension=n_qubits,
        reps=reps,
        entanglement=entanglement
    )


def fit_quantum_angle_scaler(X_train_ae_scaled: np.ndarray) -> Tuple[MinMaxScaler, np.ndarray]:
    """
    Fit MinMaxScaler to map 4D latent features to [0, pi] on TRAIN ONLY.
    """
    scaler = MinMaxScaler(feature_range=(0.0, np.pi))
    scaler.fit(X_train_ae_scaled)
    X_scaled = np.clip(scaler.transform(X_train_ae_scaled), 0.0, np.pi).astype(np.float32)
    return scaler, X_scaled


def transform_quantum_angles(X: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """
    Transform and clip features to [0, pi] using pre-fitted scaler.
    """
    scaled = scaler.transform(X)
    return np.clip(scaled, 0.0, np.pi).astype(np.float32)


def select_record_balanced_indices(
    y: np.ndarray,
    records: np.ndarray,
    n_per_class: int = 500,
    seed: int = 42
) -> np.ndarray:
    """
    Select a record-balanced subset of training samples (e.g. 500 Normal, 500 Apnea)
    to keep quantum statevector simulation computationally tractable and balanced.
    """
    y = np.asarray(y)
    records = np.asarray(records)
    rng = np.random.default_rng(seed)
    selected = []

    for cls in [0, 1]:
        cls_records = np.unique(records[y == cls])
        cls_records = cls_records.copy()
        rng.shuffle(cls_records)

        pools = {
            rec: rng.permutation(np.where((y == cls) & (records == rec))[0])
            for rec in cls_records
        }
        chosen = []

        while len(chosen) < n_per_class and any(p.size for p in pools.values()):
            for rec in cls_records:
                if pools[rec].size and len(chosen) < n_per_class:
                    chosen.append(int(pools[rec][0]))
                    pools[rec] = pools[rec][1:]

        if len(chosen) < n_per_class:
            # Fallback if specific per-record pool is small
            all_cls_idx = np.where(y == cls)[0]
            if len(all_cls_idx) >= n_per_class:
                chosen = rng.choice(all_cls_idx, size=n_per_class, replace=False).tolist()
            else:
                chosen = rng.choice(all_cls_idx, size=n_per_class, replace=True).tolist()

        selected.extend(chosen)

    selected = np.asarray(selected, dtype=int)
    rng.shuffle(selected)
    return selected


def compute_quantum_statevectors(
    X_angles: np.ndarray,
    feature_map,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> np.ndarray:
    """
    Compute 2^4 = 16-dimensional complex statevectors for each sample vector.
    """
    X_angles = np.asarray(X_angles, dtype=float)
    dim = 2 ** feature_map.num_qubits
    n_samples = len(X_angles)
    states = np.empty((n_samples, dim), dtype=np.complex128)

    for i in range(n_samples):
        circuit = feature_map.assign_parameters(X_angles[i])
        states[i] = Statevector.from_instruction(circuit).data
        if progress_callback and (i % 50 == 0 or i == n_samples - 1):
            progress_callback(i + 1, n_samples)

    return states


def compute_fidelity_kernel(states_A: np.ndarray, states_B: np.ndarray) -> np.ndarray:
    """
    Compute Quantum Fidelity Kernel matrix:
    K_ij = |<psi_i | phi_j>|^2 = |states_A_i * conj(states_B_j)|^2
    """
    return np.abs(states_A.conj() @ states_B.T) ** 2


def analyze_kernel_class_separation(
    K_train: np.ndarray,
    y_train: np.ndarray
) -> Dict[str, float]:
    """
    Compute same-class vs cross-class similarity and kernel gap.
    """
    y = np.asarray(y_train)
    off_diag = ~np.eye(len(y), dtype=bool)
    same_mask = (y[:, None] == y[None, :]) & off_diag
    cross_mask = (y[:, None] != y[None, :])

    same_mean = float(K_train[same_mask].mean()) if same_mask.any() else 0.0
    cross_mean = float(K_train[cross_mask].mean()) if cross_mask.any() else 0.0
    same_median = float(np.median(K_train[same_mask])) if same_mask.any() else 0.0
    cross_median = float(np.median(K_train[cross_mask])) if cross_mask.any() else 0.0

    return {
        "same_class_mean": same_mean,
        "cross_class_mean": cross_mean,
        "kernel_gap_mean": same_mean - cross_mean,
        "same_class_median": same_median,
        "cross_class_median": cross_median,
        "kernel_gap_median": same_median - cross_median,
    }


def tune_and_train_qsvm(
    K_train: np.ndarray,
    y_q_train: np.ndarray,
    K_val_svc: np.ndarray,
    y_val: np.ndarray,
    K_test_svc: np.ndarray,
    y_test: np.ndarray,
    c_grid: List[float] = [0.1, 1.0, 10.0, 100.0],
    seed: int = 42
) -> Tuple[SVC, float, pd.DataFrame, Dict[str, Any]]:
    """
    Tune QSVM C parameter on the VALIDATION kernel matrix only.
    Then train the final QSVM and return validation & test scores.
    """
    tuning_rows = []
    best_auc = -1.0
    best_C = 1.0

    for C in c_grid:
        m = SVC(
            kernel="precomputed",
            C=C,
            probability=False,
            random_state=seed,
        )
        t0 = time.perf_counter()
        m.fit(K_train, y_q_train)
        fit_t = time.perf_counter() - t0

        v_scores = m.decision_function(K_val_svc)
        metrics = evaluate_predictions(
            y_val, v_scores, "validation",
            threshold=0.0,
            model_name="QSVM",
            representation="4D_Autoencoder_QuantumKernel"
        )
        metrics["C"] = C
        metrics["fit_time_sec"] = fit_t
        tuning_rows.append(metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_C = C

    tuning_df = pd.DataFrame(tuning_rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

    # Final QSVM
    final_qsvm = SVC(
        kernel="precomputed",
        C=best_C,
        probability=False,
        random_state=seed
    )
    t0 = time.perf_counter()
    final_qsvm.fit(K_train, y_q_train)
    final_fit_time = time.perf_counter() - t0

    val_scores = final_qsvm.decision_function(K_val_svc)
    test_scores = final_qsvm.decision_function(K_test_svc)

    results = {
        "final_qsvm": final_qsvm,
        "best_C": best_C,
        "fit_time_sec": final_fit_time,
        "val_scores": val_scores,
        "test_scores": test_scores,
    }

    return final_qsvm, best_C, tuning_df, results
