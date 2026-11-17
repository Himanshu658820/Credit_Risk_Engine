"""Validation for already-engineered model inputs, not raw application forms."""
import numpy as np
import pandas as pd


def validate_features(data, feature_names):
    frame = pd.DataFrame([data]) if isinstance(data, dict) else data.copy()
    if frame.empty or frame.columns.duplicated().any():
        raise ValueError("Expected nonempty inputs with unique column names")
    missing = sorted(set(feature_names) - set(frame.columns))
    extra = sorted(set(frame.columns) - set(feature_names))
    if missing or extra:
        raise ValueError(f"Missing {len(missing)} required engineered features: {missing}; unexpected: {extra}")
    frame = frame.loc[:, feature_names]
    if any(not pd.api.types.is_numeric_dtype(t) and not pd.api.types.is_bool_dtype(t)
           for t in frame.dtypes):
        raise ValueError("Engineered features must be numeric or boolean")
    if np.isinf(frame.to_numpy(dtype=float)).any():
        raise ValueError("Infinite feature values are not allowed")
    # NaN represents unavailable history and is supported by LightGBM.
    return frame


def bounded_values(values, name, minimum=0, maximum=None):
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all() or (values < minimum).any():
        raise ValueError(f"{name} must be finite and >= {minimum}")
    if maximum is not None and (values > maximum).any():
        raise ValueError(f"{name} must be <= {maximum}")
    return values
