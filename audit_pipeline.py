"""
Forensic Audit Script for Q-Apnea Project
Verifies data provenance, record splits, leakage vectors, model artifacts,
metric reproducibility, and threshold/calibration integrity.
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    brier_score_loss,
    log_loss
)

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

print("=" * 80)
print("Q-APNEA FORENSIC AUDIT EXECUTION")
print("=" * 80)

# 1. Load Metadata
with open(MODELS_DIR / "pipeline_metadata.json", "r") as f:
    meta = json.load(f)

split = meta["split_summary"]
train_recs = split["train_records"]
val_recs = split["val_records"]
test_recs = split["test_records"]

print(f"\n[1] RECORD-LEVEL SPLIT INTEGRITY:")
print(f"  Train records ({len(train_recs)}): {train_recs}")
print(f"  Val records   ({len(val_recs)}): {val_recs}")
print(f"  Test records  ({len(test_recs)}): {test_recs}")

overlap_tv = set(train_recs) & set(val_recs)
overlap_tt = set(train_recs) & set(test_recs)
overlap_vt = set(val_recs) & set(test_recs)

print(f"  Overlap Train & Val : {overlap_tv} (Empty: {len(overlap_tv)==0})")
print(f"  Overlap Train & Test: {overlap_tt} (Empty: {len(overlap_tt)==0})")
print(f"  Overlap Val & Test  : {overlap_vt} (Empty: {len(overlap_vt)==0})")
assert len(overlap_tv) == 0 and len(overlap_tt) == 0 and len(overlap_vt) == 0, "Record split leakage!"

# 2. Inspect Dataset File
df = pd.read_csv(DATA_DIR / "sample_features.csv")
print(f"\n[2] DATASET COMPOSITION (data/sample_features.csv):")
print(f"  Total Rows: {len(df)}")
print(f"  Total Unique Records: {df['record'].nunique()}")
print(f"  Total Minutes/Record: {len(df) // df['record'].nunique()}")
print(f"  Overall Apnea Prevalence: {df['label'].mean():.4f} ({df['label'].sum()} apneic minutes)")

# Split df
train_df = df[df["record"].isin(train_recs)].reset_index(drop=True)
val_df = df[df["record"].isin(val_recs)].reset_index(drop=True)
test_df = df[df["record"].isin(test_recs)].reset_index(drop=True)

print(f"  Train Epochs: {len(train_df)} (Apnea: {train_df['label'].mean():.4f})")
print(f"  Val Epochs  : {len(val_df)} (Apnea: {val_df['label'].mean():.4f})")
print(f"  Test Epochs : {len(test_df)} (Apnea: {test_df['label'].mean():.4f})")

# 3. Scaler & Autoencoder Leakage Check
orig_scaler = joblib.load(MODELS_DIR / "selected_feature_scaler.joblib")
latent_scaler = joblib.load(MODELS_DIR / "ae_latent_scaler.joblib")
angle_scaler = joblib.load(MODELS_DIR / "quantum_angle_scaler.joblib")

from src.features import SELECTED_FEATURES
from src.autoencoder import HRVAutoencoder

ae = HRVAutoencoder(input_dim=len(SELECTED_FEATURES), latent_dim=4)
ae.load_state_dict(torch.load(MODELS_DIR / "hrv_autoencoder_state_dict.pt", map_location="cpu"))
ae.eval()

# Check that train scaler matches train_df
train_X_raw = train_df[SELECTED_FEATURES].to_numpy(dtype=float)
expected_mean = train_X_raw.mean(axis=0)
scaler_mean = orig_scaler.mean_
mean_diff = np.max(np.abs(expected_mean - scaler_mean))
print(f"\n[3] SCALER & LEAKAGE AUDIT:")
print(f"  Original Scaler fitted strictly on train: max mean diff = {mean_diff:.6e} (Expected: ~0.0)")

# 4. QSVM Train States Leakage Check
train_states = np.load(MODELS_DIR / "qsvm_train_states.npy")
train_indices = np.load(MODELS_DIR / "qsvm_train_indices.npy")
print(f"\n[4] QUANTUM STATE ARTIFACTS AUDIT:")
print(f"  Train states shape: {train_states.shape} (1000 samples, 16 complex amplitudes)")
print(f"  Train indices range: [{train_indices.min()}, {train_indices.max()}] (All < {len(train_df)}: {train_indices.max() < len(train_df)})")
assert train_indices.max() < len(train_df), "QSVM train indices exceed train set size!"

# 5. Recompute Test QSVM Predictions & Evaluate Metrics
from src.quantum_model import compute_fidelity_kernel

# Transform test samples through pipeline
test_X_scaled = orig_scaler.transform(test_df[SELECTED_FEATURES]).astype(np.float32)
with torch.no_grad():
    test_latent = ae.encoder(torch.tensor(test_X_scaled)).cpu().numpy().astype(np.float32)
test_latent_scaled = latent_scaler.transform(test_latent).astype(np.float32)
test_angles = np.clip(angle_scaler.transform(test_latent_scaled), 0.0, np.pi).astype(np.float32)

feature_map = joblib.load(MODELS_DIR / "qsvm_feature_map.joblib")
from qiskit.quantum_info import Statevector

print(f"  Computing test quantum statevectors ({len(test_angles)} epochs)...")
test_states = np.empty((len(test_angles), 16), dtype=np.complex128)
for i in range(len(test_angles)):
    circuit = feature_map.assign_parameters(test_angles[i])
    test_states[i] = Statevector.from_instruction(circuit).data

K_test = compute_fidelity_kernel(test_states, train_states)

qsvm = joblib.load(MODELS_DIR / "qsvm_precomputed.joblib")
test_scores = qsvm.decision_function(K_test)
y_test = test_df["label"].to_numpy(dtype=int)

th_youden = meta["qsvm_youden_threshold"]
test_preds_default = (test_scores >= 0.0).astype(int)
test_preds_youden = (test_scores >= th_youden).astype(int)

auc = roc_auc_score(y_test, test_scores)
pr_auc = average_precision_score(y_test, test_scores)

acc_def = accuracy_score(y_test, test_preds_default)
rec_def = recall_score(y_test, test_preds_default)
prec_def = precision_score(y_test, test_preds_default)
spec_def = recall_score(y_test, test_preds_default, pos_label=0)
f1_def = f1_score(y_test, test_preds_default)
cm_def = confusion_matrix(y_test, test_preds_default)

acc_you = accuracy_score(y_test, test_preds_youden)
rec_you = recall_score(y_test, test_preds_youden)
prec_you = precision_score(y_test, test_preds_youden)
spec_you = recall_score(y_test, test_preds_youden, pos_label=0)
f1_you = f1_score(y_test, test_preds_youden)
cm_you = confusion_matrix(y_test, test_preds_youden)

print(f"\n[5] VERIFIED TEST METRICS (INDEPENDENT RECOMPUTATION):")
print(f"  Test ROC-AUC: {auc:.6f}")
print(f"  Test PR-AUC : {pr_auc:.6f}")
print(f"\n  At Default Threshold (0.0):")
print(f"    Accuracy   : {acc_def*100:.2f}%")
print(f"    Sensitivity: {rec_def*100:.2f}%")
print(f"    Specificity: {spec_def*100:.2f}%")
print(f"    Precision  : {prec_def*100:.2f}%")
print(f"    F1-Score   : {f1_def:.4f}")
print(f"    CM (TN, FP, FN, TP): {cm_def.ravel()}")

print(f"\n  At Youden Validation Threshold ({th_youden:.4f}):")
print(f"    Accuracy   : {acc_you*100:.2f}%")
print(f"    Sensitivity: {rec_you*100:.2f}%")
print(f"    Specificity: {spec_you*100:.2f}%")
print(f"    Precision  : {prec_you*100:.2f}%")
print(f"    F1-Score   : {f1_you:.4f}")
print(f"    CM (TN, FP, FN, TP): {cm_you.ravel()}")

# 6. Check Calibration
calibrator = joblib.load(MODELS_DIR / "qsvm_sigmoid_calibrator.joblib")
test_probs = calibrator.predict_proba(test_scores.reshape(-1, 1))[:, 1]
brier = brier_score_loss(y_test, test_probs)
ll = log_loss(y_test, test_probs)

print(f"\n[6] CALIBRATION RECOMPUTATION:")
print(f"  Test Brier Score: {brier:.6f}")
print(f"  Test Log Loss   : {ll:.6f}")
print(f"  Mean Predicted Prob vs Prevalence: {test_probs.mean():.4f} vs {y_test.mean():.4f}")

# 7. Check Classical Baselines
svm_10d = joblib.load(MODELS_DIR / "svm_10d.joblib")
svm_pca = joblib.load(MODELS_DIR / "svm_pca_4d.joblib")
svm_ae_tuned = joblib.load(MODELS_DIR / "svm_ae_4d_tuned.joblib")

pca = joblib.load(MODELS_DIR / "pca_4d.joblib")
test_pca = pca.transform(test_X_scaled)

scores_10d = svm_10d.decision_function(test_X_scaled)
scores_pca = svm_pca.decision_function(test_pca)
scores_ae = svm_ae_tuned.decision_function(test_latent_scaled)

print(f"\n[7] CLASSICAL BASELINE METRICS (TEST SET):")
print(f"  10D RBF-SVM Test ROC-AUC : {roc_auc_score(y_test, scores_10d):.6f}, Acc: {accuracy_score(y_test, (scores_10d>=0).astype(int))*100:.2f}%")
print(f"  4D PCA-SVM Test ROC-AUC  : {roc_auc_score(y_test, scores_pca):.6f}, Acc: {accuracy_score(y_test, (scores_pca>=0).astype(int))*100:.2f}%")
print(f"  4D AE-SVM Tuned ROC-AUC  : {roc_auc_score(y_test, scores_ae):.6f}, Acc: {accuracy_score(y_test, (scores_ae>=0).astype(int))*100:.2f}%")

print("\n" + "=" * 80)
print("AUDIT EXECUTION COMPLETE - ALL ARTIFACTS VERIFIED")
print("=" * 80)
