"""
HR and HRV Feature Extraction Module
Implements minute-level time-domain Heart Rate (HR) and Heart Rate Variability (HRV)
feature extraction from RR intervals (ECG annotations).

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

FS_EXPECTED = 100
MIN_RR = 0.30       # Minimum physiologically valid RR interval (seconds) -> 200 bpm
MAX_RR = 2.00       # Maximum physiologically valid RR interval (seconds) -> 30 bpm
MIN_VALID_RR = 3    # Minimum number of valid RR intervals required in a 60s window

ALL_RAW_FEATURES = [
    "mean_hr",
    "median_hr",
    "std_hr",
    "min_hr",
    "max_hr",
    "mean_rr",
    "median_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "rr_iqr",
    "rr_range",
    "rr_std",
    "rr_cv",
    "rr_mad",
]

SELECTED_FEATURES = [
    "mean_hr",
    "std_hr",
    "min_hr",
    "max_hr",
    "mean_rr",
    "sdnn",
    "rmssd",
    "pnn50",
    "rr_iqr",
    "rr_range",
]


def extract_hrv_features_from_peaks(
    r_peaks: np.ndarray,
    start_sample: int,
    end_sample: int,
    fs: int = 100
) -> Optional[Dict[str, float]]:
    """
    Extract 15 HR/HRV features from R-peak sample indices within a 60-second window.
    
    Parameters:
        r_peaks: 1D array of sample indices of R peaks
        start_sample: Start sample of the 60s window
        end_sample: End sample of the 60s window
        fs: Sampling rate in Hz (default 100 Hz for PhysioNet Apnea-ECG)
    
    Returns:
        Dict of 15 features or None if insufficient valid beats.
    """
    peaks = r_peaks[(r_peaks >= start_sample) & (r_peaks < end_sample)]
    if len(peaks) < 4:
        return None

    rr_all = np.diff(peaks) / fs
    if len(rr_all) == 0:
        return None

    rr = rr_all[(rr_all >= MIN_RR) & (rr_all <= MAX_RR)]
    if len(rr) < MIN_VALID_RR:
        return None

    rr_valid_ratio = len(rr) / len(rr_all)
    hr = 60.0 / rr
    diff_rr = np.diff(rr)

    if len(diff_rr) == 0:
        return None

    mean_hr = float(np.mean(hr))
    median_hr = float(np.median(hr))
    std_hr = float(np.std(hr))
    min_hr = float(np.min(hr))
    max_hr = float(np.max(hr))

    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))

    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05) * 100.0)

    rr_iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    rr_range = float(np.max(rr) - np.min(rr))
    rr_std = float(np.std(rr))
    rr_cv = float(rr_std / mean_rr) if mean_rr > 0 else np.nan

    rr_median = float(np.median(rr))
    rr_mad = float(np.median(np.abs(rr - rr_median)))

    return {
        "mean_hr": mean_hr,
        "median_hr": median_hr,
        "std_hr": std_hr,
        "min_hr": min_hr,
        "max_hr": max_hr,
        "mean_rr": mean_rr,
        "median_rr": median_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "rr_iqr": rr_iqr,
        "rr_range": rr_range,
        "rr_std": rr_std,
        "rr_cv": rr_cv,
        "rr_mad": rr_mad,
        "num_r_peaks": len(peaks),
        "num_valid_rr": len(rr),
        "rr_valid_ratio": rr_valid_ratio,
    }


def compute_features_from_rr_intervals(rr_intervals: np.ndarray) -> Dict[str, float]:
    """
    Compute features directly from an array of RR intervals (in seconds).
    """
    rr_all = np.asarray(rr_intervals, dtype=float)
    rr = rr_all[(rr_all >= MIN_RR) & (rr_all <= MAX_RR)]
    if len(rr) < MIN_VALID_RR:
        raise ValueError(f"Too few valid RR intervals ({len(rr)} < {MIN_VALID_RR})")

    hr = 60.0 / rr
    diff_rr = np.diff(rr)
    if len(diff_rr) == 0:
        diff_rr = np.array([0.0])

    mean_hr = float(np.mean(hr))
    median_hr = float(np.median(hr))
    std_hr = float(np.std(hr))
    min_hr = float(np.min(hr))
    max_hr = float(np.max(hr))

    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))

    sdnn = float(np.std(rr, ddof=1)) if len(rr) > 1 else float(np.std(rr))
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2)))
    pnn50 = float(np.mean(np.abs(diff_rr) > 0.05) * 100.0)

    rr_iqr = float(np.percentile(rr, 75) - np.percentile(rr, 25))
    rr_range = float(np.max(rr) - np.min(rr))
    rr_std = float(np.std(rr))
    rr_cv = float(rr_std / mean_rr) if mean_rr > 0 else 0.0

    rr_median = float(np.median(rr))
    rr_mad = float(np.median(np.abs(rr - rr_median)))

    return {
        "mean_hr": mean_hr,
        "median_hr": median_hr,
        "std_hr": std_hr,
        "min_hr": min_hr,
        "max_hr": max_hr,
        "mean_rr": mean_rr,
        "median_rr": median_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "rr_iqr": rr_iqr,
        "rr_range": rr_range,
        "rr_std": rr_std,
        "rr_cv": rr_cv,
        "rr_mad": rr_mad,
    }
