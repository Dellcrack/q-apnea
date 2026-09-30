"""
End-to-End Pipeline Execution and Artifact Generator
Executes the full validated SIH 2026 pipeline:
1. Data loading / ingestion
2. Record-level splitting
3. Train-only median imputation & standardization
4. 4D PCA and 4D Autoencoder training
5. Classical RBF-SVM baselines (10D, 4D PCA, 4D AE, Tuned, Matched 1000)
6. Quantum ZZFeatureMap encoding & angle scaling to [0, pi]
7. Qiskit Statevector simulation & Fidelity Quantum Kernel
8. QSVM training and validation C-parameter tuning
9. Validation-derived decision thresholding (Youden's J & F1)
10. Sigmoid probability calibration
11. Test set evaluation & perturbation explainability
12. Artifacts export to models/ and results/

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple, Callable
import numpy as np
import pandas as pd
import torch
import joblib

from src.features import ALL_RAW_FEATURES, SELECTED_FEATURES
from src.data_loader import generate_representative_dataset
from src.preprocessing import (
    record_level_split,
    preprocess_datasets,
    compute_correlation_analysis,
    fit_pca_4d,
)
from src.autoencoder import (
    train_autoencoder,
    extract_latent_representations,
    HRVAutoencoder,
)
from src.classical_model import (
    fit_classical_rbf_svm,
    tune_ae_svm_on_validation,
    fit_matched_classical_baseline,
)
from src.quantum_model import (
    build_zz_feature_map,
    fit_quantum_angle_scaler,
    transform_quantum_angles,
    select_record_balanced_indices,
    compute_quantum_statevectors,
    compute_fidelity_kernel,
    analyze_kernel_class_separation,
    tune_and_train_qsvm,
)
from src.evaluation import (
    evaluate_predictions,
    compute_roc_pr_curves,
    search_optimal_thresholds,
)
from src.calibration import (
    fit_sigmoid_calibrator,
    predict_calibrated_probabilities,
    evaluate_calibration,
)
from src.explainability import (
    predict_single_sample_pipeline,
    compute_perturbation_explainability,
)


def run_full_pipeline(
    df: Optional[pd.DataFrame] = None,
    output_base_dir: Optional[Path] = None,
    qsvm_n_per_class: int = 500,
    ae_epochs: int = 150,
    seed: int = 42,
    progress_callback: Optional[callable] = None
) -> Dict[str, Any]:
    """
    Execute the entire hybrid ML/QML pipeline.
    """
    if output_base_dir is None:
        output_base_dir = Path.cwd()

    output_base_dir = Path(output_base_dir)
    data_dir = output_base_dir / "data"
    model_dir = output_base_dir / "models"
    result_dir = output_base_dir / "results"

    for folder in [data_dir, model_dir, result_dir]:
        folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)
    torch.manual_seed(seed)

    def notify(step_name: str, pct: float, detail: str = ""):
        if progress_callback:
            progress_callback(step_name, pct, detail)
        else:
            print(f"[{pct*100:5.1f}%] {step_name} - {detail}")

    # Step 1: Data acquisition
    notify("Dataset Preparation", 0.05, "Loading physiological dataset")
    if df is None:
        sample_path = data_dir / "sample_features.csv"
        if sample_path.exists():
            df = pd.read_csv(sample_path)
        else:
            df = generate_representative_dataset(n_records=35, minutes_per_record=480, seed=seed)
            df.to_csv(sample_path, index=False)

    # Step 2: Record-level splitting
    notify("Data Splitting", 0.10, "Splitting records without patient leakage (c05+c06 grouped)")
    train_df, val_df, test_df, split_meta = record_level_split(df, seed=seed)

    # Step 3: Preprocessing & Scaling (TRAIN ONLY)
    notify("Preprocessing & Imputation", 0.15, "Standardizing selected 10 HRV features (Train only)")
    (
        X_train_selected, X_val_selected, X_test_selected,
        y_train, y_val, y_test,
        scaler_selected, train_medians, prep_summary
    ) = preprocess_datasets(train_df, val_df, test_df, SELECTED_FEATURES)

    # Step 4: Correlation & PCA
    notify("Dimensionality Reduction", 0.20, "Fitting 4D PCA and correlation analysis")
    corr_results = compute_correlation_analysis(train_df, SELECTED_FEATURES)
    X_train_pca, X_val_pca, X_test_pca, pca_model, pca_var_df = fit_pca_4d(
        X_train_selected, X_val_selected, X_test_selected, seed=seed
    )

    # Step 5: Autoencoder Representation Learning
    notify("Autoencoder Training", 0.30, f"Training 10D->8D->4D Autoencoder for {ae_epochs} epochs")
    ae_model, ae_loss_history, ae_stats = train_autoencoder(
        X_train_selected,
        X_val_selected,
        epochs=ae_epochs,
        batch_size=128,
        seed=seed
    )

    # Step 6: AE Latent Extraction & Scaler (TRAIN ONLY)
    notify("Latent Space Extraction", 0.40, "Extracting and scaling 4D latent vectors")
    (
        X_train_ae_scaled, X_val_ae_scaled, X_test_ae_scaled,
        ae_latent_scaler, latent_stats
    ) = extract_latent_representations(
        ae_model, X_train_selected, X_val_selected, X_test_selected
    )

    # Step 7: Classical Models
    notify("Classical Baselines", 0.50, "Training 10D, 4D PCA, and 4D AE RBF-SVM baselines")
    res_10d = fit_classical_rbf_svm(
        X_train_selected, y_train, X_val_selected, y_val, X_test_selected, y_test,
        C=1.0, gamma="scale", representation="10D_HRV", model_name="10D RBF-SVM", seed=seed
    )
    res_pca = fit_classical_rbf_svm(
        X_train_pca, y_train, X_val_pca, y_val, X_test_pca, y_test,
        C=1.0, gamma="scale", representation="4D_PCA", model_name="4D PCA RBF-SVM", seed=seed
    )
    res_ae_fixed = fit_classical_rbf_svm(
        X_train_ae_scaled, y_train, X_val_ae_scaled, y_val, X_test_ae_scaled, y_test,
        C=1.0, gamma="scale", representation="4D_Autoencoder", model_name="4D AE RBF-SVM (Fixed)", seed=seed
    )

    notify("Classical Tuning", 0.55, "Tuning AE-SVM C and gamma on validation set")
    res_ae_tuned, ae_tune_df = tune_ae_svm_on_validation(
        X_train_ae_scaled, y_train, X_val_ae_scaled, y_val, X_test_ae_scaled, y_test,
        seed=seed
    )

    # Step 8: Quantum Setup & Subset
    notify("Quantum Feature Mapping", 0.60, "Configuring ZZFeatureMap and [0, pi] angle scaling")
    feature_map = build_zz_feature_map(n_qubits=4, reps=2, entanglement="linear")
    quantum_angle_scaler, X_quantum_train_all = fit_quantum_angle_scaler(X_train_ae_scaled)
    X_quantum_val = transform_quantum_angles(X_val_ae_scaled, quantum_angle_scaler)
    X_quantum_test = transform_quantum_angles(X_test_ae_scaled, quantum_angle_scaler)

    train_sample_records = train_df["record"].to_numpy()
    n_per_class = min(qsvm_n_per_class, int(min(np.sum(y_train == 0), np.sum(y_train == 1))))
    qs_train_idx = select_record_balanced_indices(
        y_train, train_sample_records, n_per_class=n_per_class, seed=seed
    )

    X_q_train = X_quantum_train_all[qs_train_idx]
    y_q_train = y_train[qs_train_idx]

    # Matched Classical Baseline (for fair comparison)
    notify("Matched Classical Baseline", 0.65, "Fitting matched 1000-sample classical baseline")
    # Need unscaled latent for matched scaler
    with torch.no_grad():
        t_tr = torch.tensor(X_train_selected, dtype=torch.float32)
        X_train_ae_unscaled = ae_model.encoder(t_tr).numpy().astype(np.float32)
        t_v = torch.tensor(X_val_selected, dtype=torch.float32)
        X_val_ae_unscaled = ae_model.encoder(t_v).numpy().astype(np.float32)
        t_te = torch.tensor(X_test_selected, dtype=torch.float32)
        X_test_ae_unscaled = ae_model.encoder(t_te).numpy().astype(np.float32)

    res_matched, matched_tune_df = fit_matched_classical_baseline(
        X_train_ae_unscaled, y_train, train_sample_records,
        X_val_ae_unscaled, y_val, X_test_ae_unscaled, y_test,
        qs_train_idx, seed=seed
    )

    # Step 9: Quantum Statevectors & Kernel Evaluation
    notify("Quantum State Simulation", 0.70, "Simulating training quantum statevectors")
    train_states = compute_quantum_statevectors(X_q_train, feature_map)
    K_q_train = compute_fidelity_kernel(train_states, train_states)

    notify("Quantum Kernel Diagnostics", 0.75, "Analyzing quantum kernel class separation")
    kernel_separation = analyze_kernel_class_separation(K_q_train, y_q_train)

    notify("Validation Quantum States", 0.80, "Evaluating Validation Quantum Kernel")
    val_states = compute_quantum_statevectors(X_quantum_val, feature_map)
    K_q_val_svc = compute_fidelity_kernel(val_states, train_states)

    notify("QSVM Hyperparameter Tuning", 0.85, "Tuning QSVM C parameter on validation kernel")
    test_states = compute_quantum_statevectors(X_quantum_test, feature_map)
    K_q_test_svc = compute_fidelity_kernel(test_states, train_states)

    final_qsvm, best_qsvm_C, qsvm_tune_df, qsvm_exec_res = tune_and_train_qsvm(
        K_q_train, y_q_train, K_q_val_svc, y_val, K_q_test_svc, y_test, seed=seed
    )

    # Step 10: Validation Threshold Selection
    notify("Decision Threshold Search", 0.88, "Selecting optimal operating threshold from validation set")
    qsvm_val_scores = qsvm_exec_res["val_scores"]
    qsvm_test_scores = qsvm_exec_res["test_scores"]

    best_youden_th, best_f1_th, qsvm_threshold_df = search_optimal_thresholds(
        y_val, qsvm_val_scores, model_name="QSVM"
    )

    # Step 11: Sigmoid Calibration
    notify("Probability Calibration", 0.92, "Fitting sigmoid calibrator on validation scores")
    calibrator = fit_sigmoid_calibrator(qsvm_val_scores, y_val, seed=seed)
    qsvm_val_probs = predict_calibrated_probabilities(calibrator, qsvm_val_scores)
    qsvm_test_probs = predict_calibrated_probabilities(calibrator, qsvm_test_scores)

    calib_metrics, calib_bins_df, calib_curve_data = evaluate_calibration(
        y_test, qsvm_test_probs, n_bins=10
    )

    # Test Evaluations at default (0.0) and validation-selected Youden threshold
    qsvm_val_eval = evaluate_predictions(
        y_val, qsvm_val_scores, "validation",
        threshold=best_youden_th,
        model_name="QSVM (Youden Thresh)",
        representation="4D_AE_QuantumKernel"
    )
    qsvm_test_eval_default = evaluate_predictions(
        y_test, qsvm_test_scores, "test",
        threshold=0.0,
        model_name="QSVM (Default Thresh 0.0)",
        representation="4D_AE_QuantumKernel"
    )
    qsvm_test_eval_youden = evaluate_predictions(
        y_test, qsvm_test_scores, "test",
        threshold=best_youden_th,
        model_name="QSVM (Youden Thresh)",
        representation="4D_AE_QuantumKernel"
    )
    qsvm_test_eval_f1 = evaluate_predictions(
        y_test, qsvm_test_scores, "test",
        threshold=best_f1_th,
        model_name="QSVM (F1 Thresh)",
        representation="4D_AE_QuantumKernel"
    )

    # Curve computations
    qsvm_curves = compute_roc_pr_curves(y_test, qsvm_test_scores)
    classical_curves = compute_roc_pr_curves(y_test, res_ae_tuned["test_scores"])

    # Step 12: Feature Sensitivity Explainability
    notify("Feature Sensitivity", 0.95, "Running perturbation explainability on test sample")
    demo_sample = X_test_selected[0]
    explainability_df = compute_perturbation_explainability(
        raw_sample=demo_sample,
        original_scaler=scaler_selected,
        train_medians=train_medians,
        autoencoder=ae_model,
        latent_scaler=ae_latent_scaler,
        quantum_angle_scaler=quantum_angle_scaler,
        feature_map=feature_map,
        trained_qsvm=final_qsvm,
        train_quantum_states=train_states,
        calibrator=calibrator,
        decision_threshold=best_youden_th,
    )

    # Step 13: Summary Comparison Table
    notify("Exporting Artifacts", 0.98, "Compiling comparison tables and saving models")
    comparison_rows = [
        res_10d["val_metrics"],
        res_10d["test_metrics"],
        res_pca["val_metrics"],
        res_pca["test_metrics"],
        res_ae_fixed["val_metrics"],
        res_ae_fixed["test_metrics"],
        res_ae_tuned["val_metrics"],
        res_ae_tuned["test_metrics"],
        res_matched["val_metrics"],
        res_matched["test_metrics"],
        qsvm_val_eval,
        qsvm_test_eval_default,
        qsvm_test_eval_youden,
        qsvm_test_eval_f1,
    ]
    comparison_df = pd.DataFrame(comparison_rows)

    # Save CSVs
    comparison_df.to_csv(result_dir / "model_comparison_all_results.csv", index=False)
    pd.DataFrame([calib_metrics]).to_csv(result_dir / "qsvm_calibration_metrics.csv", index=False)
    calib_bins_df.to_csv(result_dir / "qsvm_calibration_bins.csv", index=False)
    explainability_df.to_csv(result_dir / "qsvm_feature_perturbation_explainability.csv", index=False)
    qsvm_threshold_df.to_csv(result_dir / "qsvm_validation_threshold_analysis.csv", index=False)

    # Save Model Artifacts
    joblib.dump(scaler_selected, model_dir / "selected_feature_scaler.joblib")
    joblib.dump(ae_latent_scaler, model_dir / "ae_latent_scaler.joblib")
    joblib.dump(quantum_angle_scaler, model_dir / "quantum_angle_scaler.joblib")
    joblib.dump(pca_model, model_dir / "pca_4d.joblib")
    joblib.dump(res_10d["model"], model_dir / "svm_10d.joblib")
    joblib.dump(res_pca["model"], model_dir / "svm_pca_4d.joblib")
    joblib.dump(res_ae_fixed["model"], model_dir / "svm_ae_4d_fixed.joblib")
    joblib.dump(res_ae_tuned["model"], model_dir / "svm_ae_4d_tuned.joblib")
    joblib.dump(res_matched["model"], model_dir / "svm_matched_classical.joblib")
    joblib.dump(final_qsvm, model_dir / "qsvm_precomputed.joblib")
    joblib.dump(feature_map, model_dir / "qsvm_feature_map.joblib")
    joblib.dump(calibrator, model_dir / "qsvm_sigmoid_calibrator.joblib")

    torch.save(ae_model.state_dict(), model_dir / "hrv_autoencoder_state_dict.pt")
    np.save(model_dir / "qsvm_train_indices.npy", qs_train_idx)
    np.save(model_dir / "qsvm_train_states.npy", train_states)

    metadata = {
        "seed": seed,
        "selected_features": SELECTED_FEATURES,
        "all_raw_features": ALL_RAW_FEATURES,
        "ae_latent_dim": 4,
        "pca_latent_dim": 4,
        "qsvm_train_per_class": n_per_class,
        "qsvm_selected_C": float(best_qsvm_C),
        "best_ae_C": float(res_ae_tuned["C"]),
        "best_ae_gamma": str(res_ae_tuned["gamma"]),
        "qsvm_youden_threshold": float(best_youden_th),
        "qsvm_f1_threshold": float(best_f1_th),
        "split_summary": split_meta,
        "kernel_separation": kernel_separation,
        "calib_metrics": calib_metrics,
        "curves": {
            "qsvm": qsvm_curves,
            "classical": classical_curves,
            "calibration": calib_curve_data,
        },
        "ae_loss_history": ae_loss_history,
        "train_medians": train_medians.to_dict(),
    }

    with open(model_dir / "pipeline_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    notify("Pipeline Finished", 1.0, "All models, metrics, and calibration artifacts successfully generated.")

    return {
        "metadata": metadata,
        "comparison_df": comparison_df,
        "explainability_df": explainability_df,
        "calib_metrics": calib_metrics,
        "calib_bins_df": calib_bins_df,
        "qsvm_threshold_df": qsvm_threshold_df,
        "split_meta": split_meta,
        "models": {
            "ae_model": ae_model,
            "final_qsvm": final_qsvm,
            "calibrator": calibrator,
            "scaler_selected": scaler_selected,
            "ae_latent_scaler": ae_latent_scaler,
            "quantum_angle_scaler": quantum_angle_scaler,
            "feature_map": feature_map,
            "train_states": train_states,
        },
        "test_eval": {
            "qsvm_youden": qsvm_test_eval_youden,
            "qsvm_default": qsvm_test_eval_default,
            "classical_tuned": res_ae_tuned["test_metrics"],
            "matched_classical": res_matched["test_metrics"],
        }
    }


if __name__ == "__main__":
    print("Executing full SIH 2026 pipeline runner...")
    results = run_full_pipeline()
    print("Done! Model comparison:")
    print(results["comparison_df"][["model", "split", "threshold", "accuracy", "sensitivity", "specificity", "f1", "roc_auc", "pr_auc"]].round(4))
