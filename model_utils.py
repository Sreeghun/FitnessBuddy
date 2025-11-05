# model_utils.py
# Utilities for feature extraction, label encoding, and calorie estimation.
# SciPy removed — uses pure NumPy for stats (skew/kurtosis).

from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd  # optional (kept for compatibility with any callers)
from sklearn.preprocessing import LabelEncoder

LABEL_ENCODER_PATH = "label_encoder.joblib"

# -----------------------------
# Numeric helpers (no SciPy)
# -----------------------------
def _skew_np(x: np.ndarray) -> float:
    """Population skewness (Fisher) computed with NumPy; returns 0 if std=0."""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(((x - m) ** 3).mean() / (s ** 3 + 1e-12))

def _kurtosis_np(x: np.ndarray) -> float:
    """Excess kurtosis (Fisher; normal→0) with NumPy; returns 0 if std=0."""
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0
    return float(((x - m) ** 4).mean() / (s ** 4 + 1e-12) - 3.0)

# -----------------------------
# Feature extraction
# -----------------------------
def extract_features_from_window(window: np.ndarray) -> np.ndarray:
    """
    Extract a 48-length feature vector from a (T, C) window.
    We compute 16 features per channel and concatenate them.
      Per channel features (16):
        1  mean
        2  std
        3  median
        4  max
        5  min
        6  energy (mean square)
        7  variance
        8  peak-to-peak (ptp)
        9  p25
        10 p75
        11 IQR (p75 - p25)
        12 skew
        13 kurtosis (excess)
        14 RMS
        15 zero-crossings count
        16 mean absolute diff (|dx| mean)
    For 3 channels (ax, ay, az) → 48 features total.
    """
    if isinstance(window, pd.DataFrame):
        # prefer accelerometer triad if present
        cols = [c for c in ["ax", "ay", "az"] if c in window.columns]
        if len(cols) >= 3:
            X = window[cols[:3]].to_numpy()
        else:
            X = window.to_numpy()
    else:
        X = np.asarray(window)

    if X.ndim != 2:
        raise ValueError(f"window must be 2D (T, C); got shape {X.shape}")

    # If more than 3 channels arrive, keep the last 3 (ax, ay, az convention in your pipeline)
    if X.shape[1] > 3:
        X = X[:, -3:]

    feats = []
    T, C = X.shape
    for ch in range(C):
        x = X[:, ch].astype(np.float64)

        # basic stats
        mean = np.mean(x)
        std = np.std(x, ddof=0)
        median = np.median(x)
        xmax = np.max(x)
        xmin = np.min(x)

        # energy & variance
        energy = float(np.mean(x * x))
        var = float(np.var(x, ddof=0))

        # spread / quantiles
        ptp = float(np.ptp(x))
        p25 = float(np.percentile(x, 25))
        p75 = float(np.percentile(x, 75))
        iqr = p75 - p25

        # shape
        skew = _skew_np(x)
        kurt = _kurtosis_np(x)

        # rms
        rms = float(np.sqrt(np.mean(x * x)))

        # dynamics
        if T >= 2:
            zc = int(np.sum((x[:-1] * x[1:]) < 0))
            madiff = float(np.mean(np.abs(np.diff(x))))
        else:
            zc = 0
            madiff = 0.0

        feats.extend([
            mean, std, median, xmax, xmin,
            energy, var, ptp, p25, p75, iqr,
            skew, kurt, rms, zc, madiff
        ])

    return np.asarray(feats, dtype=np.float32)

def windows_to_feature_matrix(windows: list[np.ndarray]) -> np.ndarray:
    """Stack features for a list of windows into (N, F)."""
    return np.vstack([extract_features_from_window(w) for w in windows])

# -----------------------------
# Label encoder helpers
# -----------------------------
def build_label_encoder(labels: list[str]) -> LabelEncoder:
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, LABEL_ENCODER_PATH)
    return le

def load_label_encoder() -> LabelEncoder | None:
    if os.path.exists(LABEL_ENCODER_PATH):
        return joblib.load(LABEL_ENCODER_PATH)
    return None

# -----------------------------
# Calorie estimation
# -----------------------------
# Simple MET mapping; tune as needed.
MET_MAP = {
    "LAYING": 1.0,
    "SITTING": 1.3,
    "STANDING": 1.4,
    "WALKING": 3.5,
    "WALKING_UPSTAIRS": 4.0,
    "WALKING_DOWNSTAIRS": 4.0,
    "JOGGING": 7.0,
    "RUNNING": 9.0
}

def estimate_calories(activity_label: str, duration_seconds: float, weight_kg: float = 70.0) -> float:
    """
    Estimate calories using MET formula:
      calories_per_min = MET * 3.5 * weight_kg / 200
      total_calories   = calories_per_min * (duration_seconds / 60)
    """
    if not activity_label:
        activity_label = "WALKING"  # safe default
    met = MET_MAP.get(activity_label.upper(), 3.0)
    calories_per_min = met * 3.5 * weight_kg / 200.0
    return float(calories_per_min * (duration_seconds / 60.0))
