"""
Test script to verify loading of all artifacts and running inference + explainability.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
import json

base_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(base_dir))

from src.features import SELECTED_FEATURES, ALL_RAW_FEATURES
from src.autoencoder import HRVAutoencoder
from src.explainability import predict_single_sample_pipeline, compute_perturbation_explainability

print("Testing model loading...")

model_dir = base_dir / "models"
result_dir = base_dir / "results"

# Load metadata
with open(model_dir / "pipeline_metadata.json", "r") as f:
    meta = json.load(f)

print("Metadata loaded. Selected C:", meta.get("qsvm_selected_C"))

# Load scalers
orig_scaler = joblib.load(model_dir / "selected_feature_scaler.joblib")
latent_scaler = joblib.load(model_dir / "ae_latent_scaler.joblib")
q_angle_scaler = joblib.load(model_dir / "quantum_angle_scaler.joblib")
qsvm = joblib.load(model_dir / "qsvm_precomputed.joblib")
feature_map = joblib.load(model_dir / "qsvm_feature_map.joblib")
calibrator = joblib.load(model_dir / "qsvm_sigmoid_calibrator.joblib")
train_states = np.load(model_dir / "qsvm_train_states.npy")

# Load PyTorch Autoencoder
ae = HRVAutoencoder(input_dim=len(SELECTED_FEATURES), latent_dim=4)
ae.load_state_dict(torch.load(model_dir / "hrv_autoencoder_state_dict.pt", map_location="cpu"))
ae.eval()

print("All models and scalers loaded successfully!")

# Test single sample prediction
sample_normal = np.array([72.0, 4.5, 60.0, 85.0, 0.833, 0.052, 0.041, 18.5, 0.065, 0.28])
sample_apnea = np.array([88.5, 16.2, 51.0, 115.0, 0.678, 0.145, 0.112, 42.0, 0.180, 0.72])

train_medians = pd.Series(dict(zip(SELECTED_FEATURES, [75.0]*10)))

pred_norm = predict_single_sample_pipeline(
    raw_selected_features=sample_normal,
    original_scaler=orig_scaler,
    train_medians=train_medians,
    autoencoder=ae,
    latent_scaler=latent_scaler,
    quantum_angle_scaler=q_angle_scaler,
    feature_map=feature_map,
    trained_qsvm=qsvm,
    train_quantum_states=train_states,
    calibrator=calibrator,
    decision_threshold=meta.get("qsvm_youden_threshold", 0.0),
    device=torch.device("cpu")
)

print("Normal sample prediction:", pred_norm)

pred_apnea = predict_single_sample_pipeline(
    raw_selected_features=sample_apnea,
    original_scaler=orig_scaler,
    train_medians=train_medians,
    autoencoder=ae,
    latent_scaler=latent_scaler,
    quantum_angle_scaler=q_angle_scaler,
    feature_map=feature_map,
    trained_qsvm=qsvm,
    train_quantum_states=train_states,
    calibrator=calibrator,
    decision_threshold=meta.get("qsvm_youden_threshold", 0.0),
    device=torch.device("cpu")
)

print("Apnea sample prediction:", pred_apnea)

# Test explainability
exp_df = compute_perturbation_explainability(
    raw_sample=sample_apnea,
    original_scaler=orig_scaler,
    train_medians=train_medians,
    autoencoder=ae,
    latent_scaler=latent_scaler,
    quantum_angle_scaler=q_angle_scaler,
    feature_map=feature_map,
    trained_qsvm=qsvm,
    train_quantum_states=train_states,
    calibrator=calibrator,
    decision_threshold=meta.get("qsvm_youden_threshold", 0.0),
    perturbation_pct=0.10
)

print("Explainability test:\n", exp_df[["Feature", "Baseline Score", "Perturbed Score", "Influence (Delta Score)"]].head())
print("ALL TESTS PASSED!")
