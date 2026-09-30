"""
API & Server Test Script
Tests all GET and POST endpoints of the Q-Apnea server running on localhost:8000.
"""

import sys
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

def get(path):
    url = f"{BASE_URL}{path}"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

def post(path, data):
    url = f"{BASE_URL}{path}"
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as response:
        return json.loads(response.read().decode("utf-8"))

print("Testing Server Endpoints...")

# 1. /api/status
status = get("/api/status")
print("[OK] /api/status:", status["status"], status["device"])

# 2. /api/metadata
meta = get("/api/metadata")
print("[OK] /api/metadata: selected C =", meta["metadata"].get("qsvm_selected_C"))

# 3. /api/benchmarks
bench = get("/api/benchmarks")
print(f"[OK] /api/benchmarks: {len(bench['benchmarks'])} model benchmark rows")

# 4. /api/calibration
cal = get("/api/calibration")
print(f"[OK] /api/calibration: {len(cal['bins'])} bins, {len(cal['metrics'])} metrics")

# 5. /api/thresholds
th = get("/api/thresholds")
print(f"[OK] /api/thresholds: {len(th['thresholds'])} threshold points")

# 6. /api/explainability_global
exp_g = get("/api/explainability_global")
print(f"[OK] /api/explainability_global: {len(exp_g['global_features'])} features")

# 7. /api/presets
presets = get("/api/presets")
print(f"[OK] /api/presets: {len(presets['presets'])} presets")

# 8. /api/demo_patient
patient = get("/api/demo_patient?record=a01&minutes=60")
print(f"[OK] /api/demo_patient: {len(patient['timeline'])} timeline points")

# 9. /api/quantum_info
q_info = get("/api/quantum_info")
print(f"[OK] /api/quantum_info: {q_info['qubits']} qubits, {q_info['feature_map']}")

# 10. /api/predict (POST)
pred = post("/api/predict", {
    "features": presets["presets"][1]["features"], # Severe Apnea Preset
    "threshold": -0.6130
})
print(f"[OK] /api/predict: prediction={pred['prediction_label']}, prob={pred['probability_percent']}%, basis_states={len(pred['basis_states_16'])}")

# 11. /api/explain (POST)
exp = post("/api/explain", {
    "features": presets["presets"][1]["features"],
    "threshold": -0.6130,
    "perturbation_pct": 0.10
})
print(f"[OK] /api/explain: {len(exp['explanations'])} feature explanations")

# 12. /api/batch_predict (POST)
batch = post("/api/batch_predict", {
    "records": patient["timeline"],
    "threshold": -0.6130
})
print(f"[OK] /api/batch_predict: AHI={batch['summary']['ahi_score']}, severity={batch['summary']['ahi_severity']}")

print("\n" + "="*50)
print("ALL 12 ENDPOINTS PASSED PERFECTLY!")
print("="*50)
