"""
Data Visualization and Plotting Utilities Module
Provides interactive Plotly charts and publication-grade visual styling for
the SIH 2026 Hybrid Quantum Machine Learning prototype.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional


# Design theme color palette
COLORS = {
    "primary": "#4F46E5",       # Indigo / Quantum
    "secondary": "#06B6D4",     # Cyan
    "accent": "#8B5CF6",        # Purple
    "success": "#10B981",       # Emerald Green
    "warning": "#F59E0B",       # Amber
    "danger": "#EF4444",        # Rose / Red
    "background": "#0F172A",    # Dark Slate
    "card_bg": "#1E293B",       # Slate 800
    "text": "#F8FAFC",          # Slate 50
    "text_muted": "#94A3B8",    # Slate 400
    "grid": "#334155",          # Slate 700
}


def create_confusion_matrix_plot(tn: int, fp: int, fn: int, tp: int, title: str = "Confusion Matrix"):
    """
    Generate an interactive Plotly heatmap for the Confusion Matrix.
    """
    z = [[tn, fp], [fn, tp]]
    x_labels = ["Predicted Normal (0)", "Predicted Apnea (1)"]
    y_labels = ["Actual Normal (0)", "Actual Apnea (1)"]

    annotations = []
    total = tn + fp + fn + tp
    for i in range(2):
        for j in range(2):
            val = z[i][j]
            pct = (val / total * 100) if total > 0 else 0
            annotations.append(
                dict(
                    x=x_labels[j],
                    y=y_labels[i],
                    text=f"<b>{val}</b><br><span style='font-size:11px; opacity:0.8;'>({pct:.1f}%)</span>",
                    showarrow=False,
                    font=dict(color="#FFFFFF", size=15),
                )
            )

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=x_labels,
            y=y_labels,
            colorscale=[[0, "#1E293B"], [0.5, "#4338CA"], [1, "#06B6D4"]],
            showscale=False,
        )
    )

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, color="#F8FAFC")),
        annotations=annotations,
        margin=dict(l=40, r=40, t=50, b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        height=320,
    )
    return fig


def create_roc_curve_plot(curve_dict: Dict[str, Any], title: str = "ROC Curve (Receiver Operating Characteristic)"):
    """
    Generate interactive ROC curve plot.
    """
    fpr = curve_dict["fpr"]
    tpr = curve_dict["tpr"]
    auc = curve_dict.get("auc", 0.5)

    fig = go.Figure()

    # Model curve
    fig.add_trace(
        go.Scatter(
            x=fpr,
            y=tpr,
            mode="lines",
            name=f"ROC (AUC = {auc:.4f})",
            line=dict(color="#06B6D4", width=3),
            fill="tozeroy",
            fillcolor="rgba(6, 182, 212, 0.15)",
        )
    )

    # Random baseline
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Chance Baseline (AUC = 0.50)",
            line=dict(color="#64748B", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> — AUC = {auc:.4f}", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="False Positive Rate (1 - Specificity)", gridcolor="#334155", range=[-0.02, 1.02]),
        yaxis=dict(title="True Positive Rate (Sensitivity)", gridcolor="#334155", range=[-0.02, 1.02]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(15,23,42,0.8)"),
        margin=dict(l=50, r=30, t=50, b=50),
        height=360,
    )
    return fig


def create_pr_curve_plot(curve_dict: Dict[str, Any], prevalence: float = 0.35, title: str = "Precision-Recall Curve"):
    """
    Generate interactive Precision-Recall curve plot.
    """
    precision = curve_dict["precision"]
    recall = curve_dict["recall"]
    auc = curve_dict.get("auc", 0.0)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name=f"PR Curve (PR-AUC = {auc:.4f})",
            line=dict(color="#8B5CF6", width=3),
            fill="tozeroy",
            fillcolor="rgba(139, 92, 246, 0.15)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[prevalence, prevalence],
            mode="lines",
            name=f"Prevalence Baseline ({prevalence:.2f})",
            line=dict(color="#64748B", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> — PR-AUC = {auc:.4f}", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Recall (Sensitivity)", gridcolor="#334155", range=[-0.02, 1.02]),
        yaxis=dict(title="Precision (PPV)", gridcolor="#334155", range=[-0.02, 1.02]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        legend=dict(x=0.55, y=0.15, bgcolor="rgba(15,23,42,0.8)"),
        margin=dict(l=50, r=30, t=50, b=50),
        height=360,
    )
    return fig


def create_calibration_curve_plot(curve_dict: Dict[str, Any], brier_score: float, title: str = "Probability Calibration Reliability Diagram"):
    """
    Generate interactive Calibration curve plot.
    """
    prob_pred = curve_dict["prob_pred"]
    prob_true = curve_dict["prob_true"]

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name=f"QSVM Calibrated (Brier = {brier_score:.4f})",
            line=dict(color="#10B981", width=3),
            marker=dict(size=9, color="#10B981", symbol="circle"),
        )
    )

    # Perfectly calibrated 45-degree line
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(color="#64748B", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        title=dict(text=f"<b>{title}</b> — Brier Score: {brier_score:.4f}", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Mean Predicted Probability", gridcolor="#334155", range=[-0.02, 1.02]),
        yaxis=dict(title="Observed Apnea Frequency", gridcolor="#334155", range=[-0.02, 1.02]),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        legend=dict(x=0.05, y=0.90, bgcolor="rgba(15,23,42,0.8)"),
        margin=dict(l=50, r=30, t=50, b=50),
        height=360,
    )
    return fig


def create_autoencoder_loss_plot(loss_history: List[float]):
    """
    Plot autoencoder reconstruction MSE loss across epochs.
    """
    epochs = list(range(1, len(loss_history) + 1))
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=epochs,
            y=loss_history,
            mode="lines",
            name="Train MSE Loss",
            line=dict(color="#4F46E5", width=2.5),
        )
    )

    fig.update_layout(
        title=dict(text="<b>Autoencoder Reconstruction Loss (MSE)</b>", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Epoch", gridcolor="#334155"),
        yaxis=dict(title="MSE Loss", gridcolor="#334155", type="log"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=50, r=30, t=50, b=50),
        height=300,
    )
    return fig


def create_explainability_bar_plot(explainability_df: pd.DataFrame, top_n: int = 10):
    """
    Horizontal bar chart for perturbation-based feature sensitivity.
    """
    df = explainability_df.head(top_n).sort_values("Absolute Influence", ascending=True)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            y=df["Feature"],
            x=df["Absolute Influence"],
            orientation="h",
            marker=dict(
                color=df["Influence (Delta Score)"].apply(lambda v: "#EF4444" if v > 0 else "#10B981"),
                line=dict(color="#FFFFFF", width=0.5),
            ),
            text=df["Influence (Delta Score)"].apply(lambda v: f"{v:+.4f}"),
            textposition="auto",
        )
    )

    fig.update_layout(
        title=dict(text="<b>Feature Sensitivity (Perturbation Influence on QSVM Decision Score)</b>", font=dict(size=15, color="#F8FAFC")),
        xaxis=dict(title="Absolute Change in Decision Score", gridcolor="#334155"),
        yaxis=dict(title="HRV Feature", gridcolor="#334155"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E2E8F0"),
        margin=dict(l=100, r=40, t=50, b=50),
        height=360,
    )
    return fig
