"""
Isotonic Regression calibration.

Gradient-boosted models rank-order well but are usually miscalibrated in
absolute probability terms (e.g. predicting 0.8 when only 60% of loans
in that bucket actually default). Isotonic Regression fits a monotonic
step function mapping raw PD -> calibrated PD such that calibrated PD
matches the observed default rate.

Fit on OOF predictions only (never on training-set-in-sample predictions,
which would be overconfident and give a falsely optimistic calibration).
"""

import pickle
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from . import config


def fit_calibrator(oof_df: pd.DataFrame) -> IsotonicRegression:
    """
    oof_df must have columns: TARGET, PD_RAW (as produced by train.py's
    export_artifacts). Returns a fitted IsotonicRegression.
    """
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
    calibrator.fit(oof_df["PD_RAW"].values, oof_df["TARGET"].values)
    return calibrator


def apply_calibration(raw_pd, calibrator: IsotonicRegression):
    """Vectorized: raw_pd can be a scalar, list, or numpy array."""
    raw_pd = np.asarray(raw_pd, dtype=float)
    return calibrator.predict(raw_pd)


def save_calibrator(calibrator, path=None):
    path = path or config.CALIBRATOR_PATH
    with open(path, "wb") as f:
        pickle.dump(calibrator, f)
    print(f"Saved calibrator: {path}")


def load_calibrator(path=None):
    path = path or config.CALIBRATOR_PATH
    with open(path, "rb") as f:
        return pickle.load(f)


def evaluate_calibration(oof_df: pd.DataFrame, calibrator: IsotonicRegression, n_bins: int = 15):
    """Returns a bin-level comparison of raw vs calibrated PD against the
    actual observed default rate — useful for a before/after sanity check."""
    calibrated = apply_calibration(oof_df["PD_RAW"].values, calibrator)
    df = oof_df.copy()
    df["PD_CALIBRATED"] = calibrated
    df["bin"] = pd.qcut(df["PD_RAW"], q=n_bins, duplicates="drop")

    summary = df.groupby("bin", observed=True).agg(
        n=("TARGET", "size"),
        actual_default_rate=("TARGET", "mean"),
        mean_pd_raw=("PD_RAW", "mean"),
        mean_pd_calibrated=("PD_CALIBRATED", "mean"),
    ).reset_index()
    return summary


if __name__ == "__main__":
    oof_df = pd.read_csv(config.OOF_PREDICTIONS_PATH)
    calibrator = fit_calibrator(oof_df)
    save_calibrator(calibrator)

    summary = evaluate_calibration(oof_df, calibrator)
    print("\nCalibration check (raw vs calibrated PD vs actual default rate, per decile):")
    print(summary.to_string(index=False))
