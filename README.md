# Q-Apnea: Hybrid Quantum-Classical Sleep Apnea Risk Detection & Explainability Platform

> **SIH 2026 Innovation Challenge — Problem Statement 26139**  
> *Hybrid Quantum Machine Learning Platform for Early Disease Detection*  
> **Research Prototype** — Demonstrating 10D ECG Heart Rate Variability (HRV) feature extraction, 4D Neural Autoencoder Latent Space compression, 4-Qubit Quantum Support Vector Machines (`ZZFeatureMap`), Platt Sigmoid Probability Calibration, and Perturbation Sensitivity Explainability.

---

## 🔬 Scientific Summary & Key Verified Metrics

All performance metrics below have been forensically audited and verified against the held-out test set ($N = 2,880$ 1-minute epochs across 6 completely untouched records):

- **QSVM Test ROC-AUC**: **`0.9999`** (`0.999996` recomputed on test kernel matrix).
- **QSVM Test PR-AUC**: **`0.9999`** (`0.999988` recomputed).
- **Test Performance at Default Margin ($\tau = 0.0000$)**:
  - **Accuracy**: **`99.86%`** (2,876 / 2,880 correct classifications).
  - **Sensitivity (Recall)**: **`100.0%`** (768 / 768 apneic epochs detected; 0 False Negatives).
  - **Specificity**: **`99.81%`** (2,108 / 2,112 normal epochs correct; 4 False Positives).
  - **Precision**: **`99.48%`** | **F1-Score**: **`0.9974`**.
- **Test Performance at Validation Youden's J Threshold ($\tau^* = -0.6130$)**:
  - **Accuracy**: **`99.34%`** | **Sensitivity**: **`100.0%`** | **Specificity**: **`99.10%`** | **F1-Score**: **`0.9878`**.
- **Latent Compression**: **`10D Standardized HRV → 4D PyTorch Autoencoder → 4-Qubit Quantum Hilbert Space`** ($\mathbb{C}^{16}$).
- **Quantum Feature Map**: 4-qubit `ZZFeatureMap` ($H^{\otimes 4}$, $R_Z(2\theta_j)$, $R_{ZZ}$ 2-repetition linear entanglement) evaluated via Qiskit Aer statevector simulation.
- **Probability Calibration**: Platt Sigmoid scaling fitted strictly on validation scores ($\text{Brier} = 0.001429$, $\text{Log Loss} = 0.007453$, $\text{ECE} \approx 0.0022$).

> **Disclaimer**: *Q-Apnea is a research prototype developed for academic and algorithmic benchmarking under SIH 2026. It is not an FDA/CE-cleared clinical medical device and is not intended for standalone medical diagnosis.*

---

## 📊 Verified Model Benchmark Comparison

Evaluated on the held-out test set ($N = 2,880$ 1-minute epochs):

| Model Architecture | Input Representation | Operating Threshold | Test ROC-AUC | Test PR-AUC | Test Accuracy | Sensitivity | Specificity | F1-Score | Confusion Matrix `[TN, FP, FN, TP]` |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **4-Qubit QSVM (Default)** | **4D Autoencoder + ZZFeatureMap** | **$\tau = 0.0000$** | **0.9999** | **0.9999** | **99.86%** | **100.0%** | **99.81%** | **0.9974** | `[2108, 4, 0, 768]` |
| **4-Qubit QSVM (Youden $\tau^*$)** | **4D Autoencoder + ZZFeatureMap** | **$\tau = -0.6130$** | **0.9999** | **0.9999** | **99.34%** | **100.0%** | **99.10%** | **0.9878** | `[2093, 19, 0, 768]` |
| Classical RBF-SVM | 10D Standardized Features | $\tau = 0.0000$ | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | `[2112, 0, 0, 768]` |
| Classical AE-SVM (Tuned) | 4D Autoencoder Latent | $\tau = 0.0000$ | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | `[2112, 0, 0, 768]` |
| Classical PCA-SVM | 4D PCA Projection | $\tau = 0.0000$ | 1.0000 | 1.0000 | 100.0% | 100.0% | 100.0% | 1.0000 | `[2112, 0, 0, 768]` |

---

## 🛡️ Leakage Audit & Record Split Integrity

Strict **record-level grouping** prevents data leakage between patients:

- **Train Records (24)**: `a01`, `a02`, `a04`, `a05`, `a08`, `a09`, `a10`, `a11`, `a12`, `a13`, `a14`, `a15`, `a17`, `a18`, `b01`, `b02`, `b03`, `b04`, `c01`, `c02`, `c05`, `c06`, `c09`, `c10` (11,520 epochs)
- **Validation Records (5)**: `a06`, `a16`, `a20`, `b05`, `c08` (2,400 epochs)
- **Test Records (6)**: `a03`, `a07`, `a19`, `c03`, `c04`, `c07` (2,880 epochs)

$$\text{Train} \cap \text{Validation} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Validation} \cap \text{Test} = \emptyset$$

*Note: Identical subject records `c05` and `c06` are strictly co-located in the Training set to eliminate cross-split patient identity leakage.*

---

## 📁 Repository Directory Structure

```
Q_Website/
│
├── data/                       # Physiological feature datasets
│   └── sample_features.csv     # 16,800 1-min epochs (35 records × 480 mins)
│
├── models/                     # Saved model artifacts & scalers
│   ├── hrv_autoencoder_state_dict.pt  # PyTorch 10D -> 4D Autoencoder weights
│   ├── selected_feature_scaler.joblib # 10D StandardScaler (Train fitted)
│   ├── ae_latent_scaler.joblib        # 4D Latent StandardScaler
│   ├── quantum_angle_scaler.joblib    # [0, π] MinMaxScaler
│   ├── qsvm_feature_map.joblib        # Qiskit ZZFeatureMap
│   ├── qsvm_precomputed.joblib        # Precomputed Kernel QSVM (C=1.0)
│   ├── qsvm_train_states.npy          # 1,000 Balanced Quantum Statevectors
│   ├── qsvm_sigmoid_calibrator.joblib # Platt Sigmoid Logistic Regressor
│   └── pipeline_metadata.json         # Training metadata & frozen parameters
│
├── results/                    # Validated experiment outputs & reports
│   ├── final_audit_report.md          # Formal scientific audit report
│   ├── model_comparison_all_results.csv
│   ├── qsvm_calibration_bins.csv
│   ├── qsvm_calibration_metrics.csv
│   ├── qsvm_validation_threshold_analysis.csv
│   └── qsvm_feature_perturbation_explainability.csv
│
├── src/                        # Core Python Pipeline Source
│   ├── features.py             # 10 Selected HRV feature definitions
│   ├── data_loader.py          # Data ingestion & validation utilities
│   ├── preprocessing.py        # Record-level splitting & standardization
│   ├── autoencoder.py          # PyTorch 10D -> 8D -> 4D Autoencoder
│   ├── classical_model.py      # Classical RBF-SVM baselines & grid tuning
│   ├── quantum_model.py        # Qiskit ZZFeatureMap & Statevector QSVM
│   ├── evaluation.py           # Multi-metric evaluation & threshold search
│   ├── calibration.py          # Platt Sigmoid calibration & Brier scores
│   ├── explainability.py       # Perturbation sensitivity analysis
│   ├── visualizations.py       # Plotly chart utilities
│   └── pipeline_runner.py      # End-to-end execution script
│
├── public/                     # Interactive Web Application Frontend
│   ├── index.html              # 5-tab responsive glassmorphic UI
│   ├── style.css               # Modern dark theme design system
│   └── app.js                  # Client controller & Chart.js renderer
│
├── server.py                   # High-performance HTTP & REST API server
├── test_inference.py           # Pipeline integration test script
├── test_api_endpoints.py       # 12-endpoint API test suite
├── audit_pipeline.py           # Forensic metric recomputation script
└── README.md                   # Project documentation
```

---

## 🚀 Running the Platform

### 1. Launch the Web Application Server

```powershell
python server.py 8000
```
Open your browser at: **`http://127.0.0.1:8000`**

### 2. Run Test & Audit Scripts

```powershell
# Verify end-to-end PyTorch + Qiskit inference & explainability
python test_inference.py

# Verify all 12 REST API endpoints
python test_api_endpoints.py

# Recompute and verify all metrics directly from saved artifacts
python audit_pipeline.py
```

---

## 🔬 Interactive Platform Features

1. **Tab 1 — Live Quantum Risk Studio**:
   - Sliders for 10 physiological HRV metrics (`mean_hr`, `sdnn`, `rmssd`, `pnn50`, etc.).
   - Demonstration presets for characteristic sleep patterns (*Healthy Normal Sleep*, *Severe OSA Event*, *Mild Hypopnea*, *Post-Apnea Arousal*, *Deep SWS*).
   - Real-time animated circular risk gauge with Platt-calibrated posterior probability.
   - 4-Stage hybrid flow display: `Standardized 10D` $\to$ `4D Autoencoder` $\to$ `Quantum Angles [0, π]` $\to$ `Fidelity Kernel`.
   - 16-state quantum statevector probability bar chart ($|0000\rangle \dots |1111\rangle$).
   - Live $+10\%$ feature perturbation sensitivity explainability chart.

2. **Tab 2 — Model Benchmark Suite**:
   - Hold-out test set leaderboard comparing QSVM against 10D RBF-SVM, 4D PCA-SVM, and 4D AE-SVM.
   - Interactive high-resolution ROC and Precision-Recall (PR) curves.
   - 2x2 Confusion Matrices with sample counts and percentages view toggle.

3. **Tab 3 — Threshold Tuning & Calibration Laboratory**:
   - Interactive decision threshold sandbox ($\tau \in [-2.0, +2.0]$) with dynamic sensitivity/specificity updates.
   - Presets for Youden's J optimal ($\tau^* = -0.6130$), F1 optimal, and default margin ($\tau = 0.0$).
   - Validation threshold sweep curve and Platt Sigmoid reliability diagram.

4. **Tab 4 — Sleep Hypnogram & Event Burden Explorer**:
   - Multi-hour continuous timeline with dual-axis heart rate & apnea risk tracking.
   - Model-estimated apnea event burden (AHI index) and severity staging.
   - Patient record selector (`a01`, `a02`, `b01`, `c01`) and CSV record upload.

5. **Tab 5 — Quantum Architecture & Mathematical Foundations**:
   - Detailed ASCII schematic of the 4-qubit `ZZFeatureMap` circuit.
   - Mathematical formulations for the Fidelity Kernel $K(x, x') = |\langle \Phi(x) | \Phi(x') \rangle|^2$, Latent Space Compression, Platt Sigmoid, and Youden's J index.

---

## 📋 SIH 2026 Deliverables Compliance

| SIH Deliverable | Description | Status | Evidence |
| :--- | :--- | :---: | :--- |
| **Deliverable 1** | Data Preprocessing & Feature Engineering | **FULLY SATISFIED** | `src/features.py`, `src/preprocessing.py` (10 HRV features, record-level splitting, train-only Z-score) |
| **Deliverable 2** | Hybrid Quantum-Classical Architecture | **FULLY SATISFIED** | `src/autoencoder.py` (PyTorch 10D $\to$ 4D) + `src/quantum_model.py` (MinMax $[0, \pi]$ $\to$ 4-Qubit ZZFeatureMap) |
| **Deliverable 3** | Quantum ML Model Implementation | **FULLY SATISFIED** | Qiskit statevector simulation, fidelity kernel matrix $K_{ij}$, precomputed SVC training & validation tuning |
| **Deliverable 4** | Prediction, Risk Scoring & Evaluation | **FULLY SATISFIED** | Platt Sigmoid calibrator, Youden's J threshold optimization, Brier score, ROC/PR curves |
| **Deliverable 5** | Software Platform & Interactive Prototype | **FULLY SATISFIED** | HTTP REST server (`server.py`) + 5-tab responsive web application (`public/index.html`, `app.js`, `style.css`) |
