# model_utils.py
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import LabelEncoder
import joblib
import os

LABEL_ENCODER_PATH = "label_encoder.joblib"

def extract_features_from_window(window):
    """
    Robust feature extractor.
    - Accepts input as DataFrame or numpy array shape (T, n_cols).
    - Detects and ignores a timestamp-like first column if present.
    - Computes 12 features per numeric sensor channel:
      [mean, std, median, max, min, energy, kurtosis, skew,
       mean_diff, var, ptp, 75th_percentile]
    Returns a 1D numpy array of length (12 * n_channels_used).
    """
    # Convert to numpy array and get column names if DataFrame
    import numpy as _np
    import pandas as _pd

    # Handle DataFrame input (preserve column names)
    col_names = None
    if hasattr(window, "columns"):
        col_names = list(window.columns)
        arr = _np.asarray(window.values, dtype=float)
    else:
        arr = _np.asarray(window, dtype=float)

    if arr.ndim != 2:
        raise ValueError(f"Expected window shape (T, n_cols), got {arr.shape}")

    T, n_cols = arr.shape

    # Heuristic: detect timestamp-like column
    #  - if first col is strictly increasing (monotonic) or named 'timestamp' or 'time', drop it
    drop_first = False
    if col_names:
        first_name = col_names[0].lower()
        if "time" in first_name or "timestamp" in first_name:
            drop_first = True

    if not drop_first:
        # check monotonic increasing numeric sequence
        first_col = arr[:, 0]
        # if values are strictly increasing and range is large relative to typical sensor noise, treat as time
        if _np.all(_np.diff(first_col) >= 0) and (first_col.max() - first_col.min() > 0.5):
            # further check: if the std of the first col is much larger than others (likely time in seconds/ms)
            if T > 5:
                drop_first = True

    if drop_first:
        arr = arr[:, 1:]
        n_cols -= 1

    # Now filter to numeric sensor columns only (all are numeric in arr)
    if n_cols < 1:
        raise ValueError("No sensor channels left after removing timestamp column.")

    feats = []
    from scipy import stats as _stats
    for ch in range(arr.shape[1]):
        x = arr[:, ch]
        feats += [
            _np.mean(x),
            _np.std(x),
            _np.median(x),
            _np.max(x),
            _np.min(x),
            _np.sum(_np.square(x)) / max(1, len(x)),  # energy
            float(_stats.kurtosis(x)),
            float(_stats.skew(x)),
            _np.mean(_np.diff(x)) if len(x) > 1 else 0.0,
            _np.var(x),
            _np.ptp(x),
            float(_np.percentile(x, 75))
        ]
    return _np.array(feats, dtype=float)



def windows_to_feature_matrix(windows):
    X = np.vstack([extract_features_from_window(w) for w in windows])
    return X

def build_label_encoder(labels):
    le = LabelEncoder()
    le.fit(labels)
    joblib.dump(le, LABEL_ENCODER_PATH)
    return le

def load_label_encoder():
    if os.path.exists(LABEL_ENCODER_PATH):
        return joblib.load(LABEL_ENCODER_PATH)
    return None

# Simple calorie estimator: uses MET mapping per activity and user weight
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

def estimate_calories(activity_label, duration_seconds, weight_kg=70.0):
    """
    Simple calorie estimate using METs:
    calories_per_min = MET * 3.5 * weight_kg / 200
    returns calories for duration_seconds
    """
    met = MET_MAP.get(activity_label.upper(), 3.0)
    calories_per_min = met * 3.5 * weight_kg / 200.0
    calories = calories_per_min * (duration_seconds / 60.0)
    return calories
