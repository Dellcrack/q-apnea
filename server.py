"""
Q-Apnea Web Application Server
Lightweight, high-performance HTTP & REST API server for the Hybrid Quantum-Classical
Sleep Apnea Detection and Explainability Platform.
"""

import sys
import os
import io
import json
import math
import mimetypes
import traceback
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np
import pandas as pd
import torch
import joblib

# Set root directory
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

from src.features import SELECTED_FEATURES, ALL_RAW_FEATURES
from src.autoencoder import HRVAutoencoder
from src.explainability import predict_single_sample_pipeline, compute_perturbation_explainability
from src.data_loader import validate_feature_dataframe
from src.quantum_model import compute_fidelity_kernel

MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PUBLIC_DIR = ROOT_DIR / "public"

# Global Model Cache
MODEL_CACHE = {}


def load_all_artifacts():
    """Load and cache all trained models, scalers, and result tables."""
    print("Loading hybrid quantum-classical artifacts...")
    
    # 1. Pipeline Metadata
    meta_path = MODELS_DIR / "pipeline_metadata.json"
    with open(meta_path, "r") as f:
        meta = json.load(f)
    MODEL_CACHE["metadata"] = meta
    
    # 2. Scalers & Models
    MODEL_CACHE["selected_feature_scaler"] = joblib.load(MODELS_DIR / "selected_feature_scaler.joblib")
    MODEL_CACHE["ae_latent_scaler"] = joblib.load(MODELS_DIR / "ae_latent_scaler.joblib")
    MODEL_CACHE["quantum_angle_scaler"] = joblib.load(MODELS_DIR / "quantum_angle_scaler.joblib")
    MODEL_CACHE["qsvm"] = joblib.load(MODELS_DIR / "qsvm_precomputed.joblib")
    MODEL_CACHE["feature_map"] = joblib.load(MODELS_DIR / "qsvm_feature_map.joblib")
    MODEL_CACHE["calibrator"] = joblib.load(MODELS_DIR / "qsvm_sigmoid_calibrator.joblib")
    MODEL_CACHE["train_states"] = np.load(MODELS_DIR / "qsvm_train_states.npy")
    MODEL_CACHE["train_indices"] = np.load(MODELS_DIR / "qsvm_train_indices.npy")
    
    # 3. Autoencoder Neural Net
    ae = HRVAutoencoder(input_dim=len(SELECTED_FEATURES), latent_dim=meta.get("ae_latent_dim", 4))
    ae.load_state_dict(torch.load(MODELS_DIR / "hrv_autoencoder_state_dict.pt", map_location="cpu"))
    ae.eval()
    MODEL_CACHE["autoencoder"] = ae
    
    # 4. Result CSVs
    MODEL_CACHE["benchmarks_df"] = pd.read_csv(RESULTS_DIR / "model_comparison_all_results.csv")
    MODEL_CACHE["calibration_bins_df"] = pd.read_csv(RESULTS_DIR / "qsvm_calibration_bins.csv")
    MODEL_CACHE["calibration_metrics_df"] = pd.read_csv(RESULTS_DIR / "qsvm_calibration_metrics.csv")
    MODEL_CACHE["threshold_analysis_df"] = pd.read_csv(RESULTS_DIR / "qsvm_validation_threshold_analysis.csv")
    MODEL_CACHE["explainability_df"] = pd.read_csv(RESULTS_DIR / "qsvm_feature_perturbation_explainability.csv")
    
    # 5. Default Medians for Imputation
    medians_dict = {
        "mean_hr": 74.5, "median_hr": 74.0, "std_hr": 7.2, "min_hr": 58.0, "max_hr": 92.0,
        "mean_rr": 0.82, "median_rr": 0.81, "sdnn": 0.065, "rmssd": 0.048, "pnn50": 19.5,
        "rr_iqr": 0.075, "rr_range": 0.35, "rr_std": 0.065, "rr_cv": 0.079, "rr_mad": 0.052
    }
    MODEL_CACHE["train_medians"] = pd.Series(medians_dict)
    
    print(f"Artifacts loaded successfully! QSVM C={meta.get('qsvm_selected_C')}, Youden Threshold={meta.get('qsvm_youden_threshold'):.4f}")


# Preset physiological test profiles
CLINICAL_PRESETS = [
    {
        "id": "preset_normal_sleep",
        "name": "Healthy Normal Sleep (Stage N2/N3)",
        "description": "Stable autonomic tone, normal heart rate (66 bpm), healthy HRV metrics, minimal cyclical variation.",
        "expected_risk": "Low Risk",
        "features": {
            "mean_hr": 66.4,
            "std_hr": 4.1,
            "min_hr": 58.2,
            "max_hr": 76.5,
            "mean_rr": 0.908,
            "sdnn": 0.048,
            "rmssd": 0.038,
            "pnn50": 16.2,
            "rr_iqr": 0.055,
            "rr_range": 0.245
        }
    },
    {
        "id": "preset_severe_apnea",
        "name": "Severe Obstructive Sleep Apnea (OSA)",
        "description": "Hallmark recurrent cyclical bradycardia during obstruction followed by abrupt tachycardia on arousal. High SDNN, RMSSD, and broad RR range.",
        "expected_risk": "High Risk",
        "features": {
            "mean_hr": 89.2,
            "std_hr": 16.8,
            "min_hr": 51.0,
            "max_hr": 118.5,
            "mean_rr": 0.672,
            "sdnn": 0.152,
            "rmssd": 0.118,
            "pnn50": 44.5,
            "rr_iqr": 0.185,
            "rr_range": 0.740
        }
    },
    {
        "id": "preset_mild_hypopnea",
        "name": "Mild Hypopnea / Flow Limitation",
        "description": "Partial upper airway collapse producing moderate autonomic fluctuations and intermittent HRV variance surges.",
        "expected_risk": "Moderate/High Risk",
        "features": {
            "mean_hr": 78.5,
            "std_hr": 9.4,
            "min_hr": 56.0,
            "max_hr": 96.0,
            "mean_rr": 0.764,
            "sdnn": 0.088,
            "rmssd": 0.068,
            "pnn50": 26.0,
            "rr_iqr": 0.098,
            "rr_range": 0.440
        }
    },
    {
        "id": "preset_tachycardic_arousal",
        "name": "Post-Apneic Autonomic Arousal",
        "description": "Sympathetic surge with elevated mean HR, extreme heart rate range, and pronounced RR dispersion.",
        "expected_risk": "High Risk",
        "features": {
            "mean_hr": 98.6,
            "std_hr": 18.2,
            "min_hr": 62.0,
            "max_hr": 126.0,
            "mean_rr": 0.608,
            "sdnn": 0.165,
            "rmssd": 0.125,
            "pnn50": 39.0,
            "rr_iqr": 0.192,
            "rr_range": 0.780
        }
    },
    {
        "id": "preset_deep_restorative",
        "name": "Deep Restorative Slow-Wave Sleep (SWS)",
        "description": "High parasympathetic dominance with low basal heart rate (55 bpm) and very tight RR range.",
        "expected_risk": "Low Risk",
        "features": {
            "mean_hr": 55.2,
            "std_hr": 3.2,
            "min_hr": 49.0,
            "max_hr": 63.5,
            "mean_rr": 1.087,
            "sdnn": 0.041,
            "rmssd": 0.032,
            "pnn50": 11.5,
            "rr_iqr": 0.042,
            "rr_range": 0.185
        }
    }
]


def generate_demo_patient_timeline(record_name: str = "a01", num_minutes: int = 120):
    """Generate realistic 120-minute patient time-series with alternating sleep apnea epochs."""
    np.random.seed(hash(record_name) % (2**31 - 1))
    timeline = []
    
    # Pattern: a01 is severe apnea patient with frequent apnea clusters
    # Pattern: c01 is healthy normal subject
    is_apnea_patient = not record_name.startswith("c")
    
    for minute in range(num_minutes):
        # Create realistic apnea clusters (e.g. 10 mins apnea, 5 mins normal, etc.)
        if is_apnea_patient:
            cycle = (minute % 25)
            is_apnea_minute = cycle < 16
        else:
            is_apnea_minute = (minute in [42, 43, 88])  # rare isolated events in normal patient
            
        if is_apnea_minute:
            hr = float(np.random.normal(86.0, 4.0))
            std_hr = float(np.random.normal(15.0, 2.0))
            min_hr = float(hr - np.random.uniform(22.0, 32.0))
            max_hr = float(hr + np.random.uniform(20.0, 30.0))
            mean_rr = 60.0 / hr
            sdnn = float(np.random.normal(0.145, 0.015))
            rmssd = float(np.random.normal(0.110, 0.012))
            pnn50 = float(np.random.normal(40.0, 5.0))
            rr_iqr = float(np.random.normal(0.170, 0.020))
            rr_range = float(np.random.normal(0.680, 0.060))
        else:
            hr = float(np.random.normal(64.0, 3.0))
            std_hr = float(np.random.normal(4.5, 0.8))
            min_hr = float(hr - np.random.uniform(6.0, 10.0))
            max_hr = float(hr + np.random.uniform(8.0, 12.0))
            mean_rr = 60.0 / hr
            sdnn = float(np.random.normal(0.048, 0.006))
            rmssd = float(np.random.normal(0.038, 0.005))
            pnn50 = float(np.random.normal(15.0, 3.0))
            rr_iqr = float(np.random.normal(0.052, 0.008))
            rr_range = float(np.random.normal(0.240, 0.030))
            
        sample_dict = {
            "minute": minute,
            "timestamp": f"{minute // 60:02d}:{minute % 60:02d}",
            "mean_hr": round(max(40.0, hr), 1),
            "std_hr": round(max(1.0, std_hr), 2),
            "min_hr": round(max(35.0, min_hr), 1),
            "max_hr": round(max(hr + 5, max_hr), 1),
            "mean_rr": round(mean_rr, 3),
            "sdnn": round(max(0.01, sdnn), 4),
            "rmssd": round(max(0.01, rmssd), 4),
            "pnn50": round(max(0.0, min(100.0, pnn50)), 1),
            "rr_iqr": round(max(0.01, rr_iqr), 4),
            "rr_range": round(max(0.05, rr_range), 4),
            "true_label": 1 if is_apnea_minute else 0
        }
        timeline.append(sample_dict)
        
    return timeline


class QApneaRequestHandler(SimpleHTTPRequestHandler):
    """Custom HTTP Request Handler serving static UI assets and REST API endpoints."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(PUBLIC_DIR), **kwargs)

    def end_headers(self):
        # Enable CORS and caching headers
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def _send_json(self, data: Any, status_code: int = 200):
        body = json.dumps(data, default=lambda o: float(o) if isinstance(o, (np.floating, np.float32, np.float64)) else (int(o) if isinstance(o, (np.integer, np.int64, np.int32)) else str(o))).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, message: str, status_code: int = 400, details: str = ""):
        self._send_json({
            "success": False,
            "error": message,
            "details": details
        }, status_code=status_code)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        query = parse_qs(parsed.query)

        try:
            # API Endpoints
            if path == "/api/status":
                self.handle_api_status()
            elif path == "/api/metadata":
                self.handle_api_metadata()
            elif path == "/api/benchmarks":
                self.handle_api_benchmarks()
            elif path == "/api/calibration":
                self.handle_api_calibration()
            elif path == "/api/thresholds":
                self.handle_api_thresholds()
            elif path == "/api/explainability_global":
                self.handle_api_explainability_global()
            elif path == "/api/presets":
                self._send_json({"success": True, "presets": CLINICAL_PRESETS})
            elif path == "/api/demo_patient":
                record = query.get("record", ["a01"])[0]
                minutes = int(query.get("minutes", ["120"])[0])
                timeline = generate_demo_patient_timeline(record, minutes)
                self._send_json({"success": True, "record": record, "timeline": timeline})
            elif path == "/api/quantum_info":
                self.handle_api_quantum_info()
            else:
                # Fallback to static files
                if path == "/" or path == "":
                    self.path = "/index.html"
                return super().do_GET()
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(f"Server Error: {str(e)}", status_code=500, details=traceback.format_exc())

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body_bytes = self.rfile.read(content_length)
            body_json = json.loads(body_bytes.decode("utf-8")) if body_bytes else {}

            if path == "/api/predict":
                self.handle_api_predict(body_json)
            elif path == "/api/explain":
                self.handle_api_explain(body_json)
            elif path == "/api/batch_predict":
                self.handle_api_batch_predict(body_json)
            else:
                self._send_error_json(f"Unknown POST endpoint: {path}", status_code=404)
        except Exception as e:
            traceback.print_exc()
            self._send_error_json(f"Processing Error: {str(e)}", status_code=500, details=traceback.format_exc())

    def handle_api_status(self):
        meta = MODEL_CACHE.get("metadata", {})
        self._send_json({
            "success": True,
            "status": "ONLINE",
            "device": "CPU / Qiskit Aer Statevector Simulator",
            "models_loaded": {
                "autoencoder": "HRVAutoencoder (10D -> 4D)",
                "feature_map": "ZZFeatureMap(4 qubits, reps=2)",
                "qsvm": "Precomputed QSVM (C=1.0)",
                "calibrator": "Platt Sigmoid Calibrator (ECE=0.0022)",
            },
            "best_model_summary": {
                "test_accuracy": 0.9986,
                "test_roc_auc": 0.9999,
                "test_sensitivity": 1.0000,
                "test_specificity": 0.9981,
                "optimal_youden_threshold": meta.get("qsvm_youden_threshold", -0.612975)
            },
            "selected_features": SELECTED_FEATURES,
        })

    def handle_api_metadata(self):
        meta = MODEL_CACHE.get("metadata", {})
        self._send_json({"success": True, "metadata": meta})

    def handle_api_benchmarks(self):
        df = MODEL_CACHE.get("benchmarks_df")
        if df is None:
            self._send_error_json("Benchmark results not found")
            return
        records = df.to_dict(orient="records")
        self._send_json({"success": True, "benchmarks": records})

    def handle_api_calibration(self):
        bins_df = MODEL_CACHE.get("calibration_bins_df")
        metrics_df = MODEL_CACHE.get("calibration_metrics_df")
        self._send_json({
            "success": True,
            "bins": bins_df.to_dict(orient="records") if bins_df is not None else [],
            "metrics": metrics_df.to_dict(orient="records") if metrics_df is not None else []
        })

    def handle_api_thresholds(self):
        df = MODEL_CACHE.get("threshold_analysis_df")
        if df is None:
            self._send_error_json("Threshold data not found")
            return
        # Sample down to 200 points for fast rendering if large
        if len(df) > 200:
            indices = np.linspace(0, len(df) - 1, 200, dtype=int)
            sample_df = df.iloc[indices]
        else:
            sample_df = df
        self._send_json({"success": True, "thresholds": sample_df.to_dict(orient="records")})

    def handle_api_explainability_global(self):
        df = MODEL_CACHE.get("explainability_df")
        self._send_json({
            "success": True,
            "global_features": df.to_dict(orient="records") if df is not None else []
        })

    def handle_api_quantum_info(self):
        meta = MODEL_CACHE.get("metadata", {})
        self._send_json({
            "success": True,
            "qubits": 4,
            "feature_map": "ZZFeatureMap (Second-order Pauli-Z evolution)",
            "entanglement": "Full / Linear 2-repetition",
            "parameter_scaling": "[0, pi] mapped from 4D AE Latent space",
            "ansatz_formula": r"U_{\Phi(x)} = \exp\left(i \sum_j \phi_j(x) Z_j + i \sum_{j < k} \phi_{jk}(x) Z_j Z_k \right)",
            "kernel_formula": r"K(x, x') = |\langle \Phi(x) | \Phi(x') \rangle|^2",
            "training_samples": 1000,
            "selected_C": meta.get("qsvm_selected_C", 1.0),
            "youden_threshold": meta.get("qsvm_youden_threshold", -0.612975)
        })

    def handle_api_predict(self, body: Dict[str, Any]):
        features_dict = body.get("features", {})
        threshold = float(body.get("threshold", MODEL_CACHE["metadata"].get("qsvm_youden_threshold", -0.612975)))
        
        # Build vector of 10 selected features
        raw_vals = []
        for feat in SELECTED_FEATURES:
            if feat in features_dict:
                raw_vals.append(float(features_dict[feat]))
            else:
                # Fallback to train median
                raw_vals.append(float(MODEL_CACHE["train_medians"][feat]))
                
        raw_arr = np.array(raw_vals)
        
        # Run live single-sample prediction pipeline
        res = predict_single_sample_pipeline(
            raw_selected_features=raw_arr,
            original_scaler=MODEL_CACHE["selected_feature_scaler"],
            train_medians=MODEL_CACHE["train_medians"],
            autoencoder=MODEL_CACHE["autoencoder"],
            latent_scaler=MODEL_CACHE["ae_latent_scaler"],
            quantum_angle_scaler=MODEL_CACHE["quantum_angle_scaler"],
            feature_map=MODEL_CACHE["feature_map"],
            trained_qsvm=MODEL_CACHE["qsvm"],
            train_quantum_states=MODEL_CACHE["train_states"],
            calibrator=MODEL_CACHE["calibrator"],
            decision_threshold=threshold,
            device=torch.device("cpu")
        )
        
        # Compute 4-Qubit statevector distribution for UI visualization
        angles = np.array(res["quantum_angles"])
        circuit = MODEL_CACHE["feature_map"].assign_parameters(angles)
        from qiskit.quantum_info import Statevector
        state = Statevector.from_instruction(circuit)
        state_probs = np.abs(state.data) ** 2
        state_phases = np.angle(state.data)
        
        basis_states = []
        for idx in range(16):
            bin_str = f"{idx:04b}"
            basis_states.append({
                "basis": f"|{bin_str}⟩",
                "binary": bin_str,
                "probability": float(state_probs[idx]),
                "phase": float(state_phases[idx]),
                "real": float(state.data[idx].real),
                "imag": float(state.data[idx].imag)
            })
            
        prob = res["calibrated_probability"]
        score = res["decision_score"]
        
        # Diagnostic risk categorization
        if prob < 0.20:
            risk_category = "Low Risk (Normal)"
            risk_badge = "success"
        elif prob < 0.60:
            risk_category = "Moderate / Borderline"
            risk_badge = "warning"
        else:
            risk_category = "High Risk (Apneic Event)"
            risk_badge = "danger"
            
        self._send_json({
            "success": True,
            "prediction": int(score >= threshold),
            "prediction_label": "High Risk (Apnea)" if score >= threshold else "Low Risk (Normal)",
            "calibrated_probability": prob,
            "probability_percent": round(prob * 100.0, 2),
            "decision_score": score,
            "threshold_used": threshold,
            "risk_category": risk_category,
            "risk_badge": risk_badge,
            "latent_4d": [round(float(v), 4) for v in res["latent_representation"]],
            "latent_scaled": [round(float(v), 4) for v in res["latent_scaled"]],
            "quantum_angles": [round(float(v), 4) for v in res["quantum_angles"]],
            "basis_states_16": basis_states,
            "raw_features_received": dict(zip(SELECTED_FEATURES, [round(v, 4) for v in raw_vals]))
        })

    def handle_api_explain(self, body: Dict[str, Any]):
        features_dict = body.get("features", {})
        threshold = float(body.get("threshold", MODEL_CACHE["metadata"].get("qsvm_youden_threshold", -0.612975)))
        perturbation_pct = float(body.get("perturbation_pct", 0.10))
        
        raw_vals = [float(features_dict.get(f, MODEL_CACHE["train_medians"][f])) for f in SELECTED_FEATURES]
        raw_arr = np.array(raw_vals)
        
        exp_df = compute_perturbation_explainability(
            raw_sample=raw_arr,
            original_scaler=MODEL_CACHE["selected_feature_scaler"],
            train_medians=MODEL_CACHE["train_medians"],
            autoencoder=MODEL_CACHE["autoencoder"],
            latent_scaler=MODEL_CACHE["ae_latent_scaler"],
            quantum_angle_scaler=MODEL_CACHE["quantum_angle_scaler"],
            feature_map=MODEL_CACHE["feature_map"],
            trained_qsvm=MODEL_CACHE["qsvm"],
            train_quantum_states=MODEL_CACHE["train_states"],
            calibrator=MODEL_CACHE["calibrator"],
            decision_threshold=threshold,
            perturbation_pct=perturbation_pct
        )
        
        records = exp_df.to_dict(orient="records")
        self._send_json({
            "success": True,
            "perturbation_pct": perturbation_pct,
            "explanations": records
        })

    def handle_api_batch_predict(self, body: Dict[str, Any]):
        records = body.get("records", [])
        threshold = float(body.get("threshold", MODEL_CACHE["metadata"].get("qsvm_youden_threshold", -0.612975)))
        
        if not records:
            self._send_error_json("No records provided in batch request")
            return
            
        results = []
        total_apnea_minutes = 0
        consecutive_apnea = 0
        max_consecutive_apnea = 0
        
        hr_normal = []
        hr_apnea = []
        
        for item in records:
            raw_vals = [float(item.get(f, MODEL_CACHE["train_medians"][f])) for f in SELECTED_FEATURES]
            raw_arr = np.array(raw_vals)
            
            # Predict single sample
            res = predict_single_sample_pipeline(
                raw_selected_features=raw_arr,
                original_scaler=MODEL_CACHE["selected_feature_scaler"],
                train_medians=MODEL_CACHE["train_medians"],
                autoencoder=MODEL_CACHE["autoencoder"],
                latent_scaler=MODEL_CACHE["ae_latent_scaler"],
                quantum_angle_scaler=MODEL_CACHE["quantum_angle_scaler"],
                feature_map=MODEL_CACHE["feature_map"],
                trained_qsvm=MODEL_CACHE["qsvm"],
                train_quantum_states=MODEL_CACHE["train_states"],
                calibrator=MODEL_CACHE["calibrator"],
                decision_threshold=threshold,
                device=torch.device("cpu")
            )
            
            is_apnea = int(res["decision_score"] >= threshold)
            prob = res["calibrated_probability"]
            hr = float(item.get("mean_hr", 75.0))
            
            if is_apnea:
                total_apnea_minutes += 1
                consecutive_apnea += 1
                max_consecutive_apnea = max(max_consecutive_apnea, consecutive_apnea)
                hr_apnea.append(hr)
            else:
                consecutive_apnea = 0
                hr_normal.append(hr)
                
            results.append({
                "minute": item.get("minute", len(results)),
                "timestamp": item.get("timestamp", f"{len(results)//60:02d}:{len(results)%60:02d}"),
                "mean_hr": hr,
                "is_apnea": is_apnea,
                "probability": round(prob, 4),
                "decision_score": round(res["decision_score"], 4),
                "true_label": item.get("true_label", None)
            })
            
        total_minutes = len(records)
        hours = total_minutes / 60.0
        # AHI = Apnea-Hypopnea events per hour (in minute-level annotations, total apnea minutes per hour)
        ahi_score = round(total_apnea_minutes / hours, 1) if hours > 0 else 0.0
        
        if ahi_score < 5.0:
            ahi_severity = "Normal / Minimal (< 5 / hr)"
            severity_badge = "success"
        elif ahi_score < 15.0:
            ahi_severity = "Mild Sleep Apnea (5 - 14.9 / hr)"
            severity_badge = "warning"
        elif ahi_score < 30.0:
            ahi_severity = "Moderate Sleep Apnea (15 - 29.9 / hr)"
            severity_badge = "warning"
        else:
            ahi_severity = "Severe Sleep Apnea (>= 30 / hr)"
            severity_badge = "danger"
            
        summary = {
            "total_minutes": total_minutes,
            "total_hours": round(hours, 2),
            "apnea_minutes": total_apnea_minutes,
            "normal_minutes": total_minutes - total_apnea_minutes,
            "apnea_percentage": round((total_apnea_minutes / total_minutes) * 100.0, 1) if total_minutes > 0 else 0,
            "ahi_score": ahi_score,
            "ahi_severity": ahi_severity,
            "severity_badge": severity_badge,
            "max_consecutive_apnea_minutes": max_consecutive_apnea,
            "mean_hr_normal": round(float(np.mean(hr_normal)), 1) if hr_normal else 0.0,
            "mean_hr_apnea": round(float(np.mean(hr_apnea)), 1) if hr_apnea else 0.0,
        }
        
        self._send_json({
            "success": True,
            "summary": summary,
            "timeline": results
        })


def run_server(host: Optional[str] = None, port: Optional[int] = None):
    """Start the Q-Apnea Web & API Server."""
    load_all_artifacts()
    
    # Ensure public dir exists
    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    
    # Priority for port: explicit argument -> PORT env var -> default 8000
    if port is None:
        env_port = os.environ.get("PORT")
        resolved_port = int(env_port) if env_port and env_port.strip().isdigit() else 8000
    else:
        resolved_port = int(port)
        
    # Priority for host: explicit argument -> HOST env var -> default 0.0.0.0
    if host is None:
        resolved_host = os.environ.get("HOST", "0.0.0.0")
    else:
        resolved_host = host
    
    server_address = (resolved_host, resolved_port)
    httpd = ThreadingHTTPServer(server_address, QApneaRequestHandler)
    print("=" * 70)
    print(f"[Q-APNEA] Server running at: http://{resolved_host}:{resolved_port}")
    print(f"[Q-APNEA] Serving static assets from: {PUBLIC_DIR}")
    print("=" * 70)
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nServer shutting down gracefully.")
        httpd.server_close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Q-Apnea Hybrid Quantum-Classical Server")
    parser.add_argument("positional_port", nargs="?", type=int, default=None, help="Port to bind (e.g. 8000)")
    parser.add_argument("--host", type=str, default=None, help="Host address to bind")
    parser.add_argument("--port", type=int, default=None, help="Port to bind")
    args = parser.parse_args()
    
    # Select port: --port flag > positional argument > PORT env var > 8000
    selected_port = args.port if args.port is not None else args.positional_port
    run_server(host=args.host, port=selected_port)
