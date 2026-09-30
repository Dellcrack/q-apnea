"""
Explainability and Feature Sensitivity Analysis Module
Implements perturbation-based feature sensitivity testing by altering
individual HR/HRV features by +10% of their standard scale and measuring
the resulting delta in the QSVM decision score.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from qiskit.quantum_info import Statevector
from typing import Dict, Any, List, Optional
from src.features import SELECTED_FEATURES
from src.quantum_model import compute_fidelity_kernel


def predict_single_sample_pipeline(
    raw_selected_features: np.ndarray,
    original_scaler: StandardScaler,
    train_medians: pd.Series,
    autoencoder: torch.nn.Module,
    latent_scaler: StandardScaler,
    quantum_angle_scaler: MinMaxScaler,
    feature_map,
    trained_qsvm,
    train_quantum_states: np.ndarray,
    calibrator,
    decision_threshold: float = 0.0,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    End-to-end inference pipeline for a single sample:
    Raw 10D -> Impute -> Standardize -> AE Encoder -> 4D Latent ->
    Latent Scale -> Quantum Angle Scale [0, pi] -> Statevector ->
    Fidelity Kernel -> QSVM Decision -> Sigmoid Calibrated Probability.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    raw = np.asarray(raw_selected_features, dtype=float).reshape(1, -1)
    if raw.shape[1] != len(SELECTED_FEATURES):
        raise ValueError(f"Expected {len(SELECTED_FEATURES)} features, got {raw.shape[1]}")

    raw_df = pd.DataFrame(raw, columns=SELECTED_FEATURES)
    if train_medians is not None:
        raw_df = raw_df.fillna(train_medians[SELECTED_FEATURES])

    scaled_orig = original_scaler.transform(raw_df).astype(np.float32)

    autoencoder.eval()
    with torch.no_grad():
        t_x = torch.tensor(scaled_orig, dtype=torch.float32, device=device)
        latent = autoencoder.encoder(t_x).cpu().numpy().astype(np.float32)

    latent_scaled = latent_scaler.transform(latent).astype(np.float32)
    quantum_angles = quantum_angle_scaler.transform(latent_scaled)
    quantum_angles = np.clip(quantum_angles, 0.0, np.pi).astype(np.float32)

    # Statevector computation
    circuit = feature_map.assign_parameters(quantum_angles[0])
    state = Statevector.from_instruction(circuit).data.reshape(1, -1)

    kernel_row = compute_fidelity_kernel(state, train_quantum_states)
    decision_score = float(trained_qsvm.decision_function(kernel_row).ravel()[0])

    calibrated_prob = float(calibrator.predict_proba(np.array([[decision_score]]))[:, 1][0])
    prediction = int(decision_score >= decision_threshold)

    return {
        "prediction": prediction,
        "prediction_label": "High Risk (Apnea)" if prediction == 1 else "Low Risk (Normal)",
        "decision_score": decision_score,
        "calibrated_probability": calibrated_prob,
        "latent_representation": latent[0].tolist(),
        "latent_scaled": latent_scaled[0].tolist(),
        "quantum_angles": quantum_angles[0].tolist(),
        "decision_threshold": decision_threshold,
    }


def compute_perturbation_explainability(
    raw_sample: np.ndarray,
    original_scaler: StandardScaler,
    train_medians: pd.Series,
    autoencoder: torch.nn.Module,
    latent_scaler: StandardScaler,
    quantum_angle_scaler: MinMaxScaler,
    feature_map,
    trained_qsvm,
    train_quantum_states: np.ndarray,
    calibrator,
    decision_threshold: float = 0.0,
    perturbation_pct: float = 0.10
) -> pd.DataFrame:
    """
    Compute feature perturbation sensitivity:
    Perturb each feature by delta = perturbation_pct * feature_scale and measure
    change in QSVM decision score.
    """
    baseline_res = predict_single_sample_pipeline(
        raw_selected_features=raw_sample,
        original_scaler=original_scaler,
        train_medians=train_medians,
        autoencoder=autoencoder,
        latent_scaler=latent_scaler,
        quantum_angle_scaler=quantum_angle_scaler,
        feature_map=feature_map,
        trained_qsvm=trained_qsvm,
        train_quantum_states=train_quantum_states,
        calibrator=calibrator,
        decision_threshold=decision_threshold,
    )
    baseline_score = baseline_res["decision_score"]

    feature_scales = original_scaler.scale_
    rows = []

    for i, feature_name in enumerate(SELECTED_FEATURES):
        perturbed = np.array(raw_sample, dtype=float).copy()
        delta = perturbation_pct * (feature_scales[i] if feature_scales[i] > 0 else 1.0)
        perturbed[i] += delta

        pert_res = predict_single_sample_pipeline(
            raw_selected_features=perturbed,
            original_scaler=original_scaler,
            train_medians=train_medians,
            autoencoder=autoencoder,
            latent_scaler=latent_scaler,
            quantum_angle_scaler=quantum_angle_scaler,
            feature_map=feature_map,
            trained_qsvm=trained_qsvm,
            train_quantum_states=train_quantum_states,
            calibrator=calibrator,
            decision_threshold=decision_threshold,
        )
        pert_score = pert_res["decision_score"]
        influence = pert_score - baseline_score

        rows.append({
            "Feature": feature_name,
            "Baseline Value": float(raw_sample[i]),
            "Perturbed Value": float(perturbed[i]),
            "Delta": float(delta),
            "Baseline Score": float(baseline_score),
            "Perturbed Score": float(pert_score),
            "Influence (Delta Score)": float(influence),
            "Absolute Influence": float(abs(influence)),
            "Direction": "Increases Risk" if influence > 0 else "Decreases Risk",
        })

    df = pd.DataFrame(rows).sort_values("Absolute Influence", ascending=False).reset_index(drop=True)
    return df
