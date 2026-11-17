"""Independent evaluation. No model or threshold selection uses test outcomes."""
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import train_test_split


def split_cohorts(frame, seed=42):
    if frame.SK_ID_CURR.duplicated().any():
        raise ValueError("Borrower IDs must be unique")
    development, test = train_test_split(frame, test_size=.2, random_state=seed,
                                         stratify=frame.TARGET)
    development, calibration = train_test_split(development, test_size=.25,
                                                random_state=seed, stratify=development.TARGET)
    return development.copy(), calibration.copy(), test.copy()


def metrics(y, probability):
    y = np.asarray(y)
    p = np.asarray(probability, dtype=float)
    if len(y) != len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid probabilities")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Evaluation requires both binary outcome classes")
    return {"roc_auc": float(roc_auc_score(y, p)),
            "average_precision": float(average_precision_score(y, p)),
            "brier_score": float(brier_score_loss(y, p)),
            "log_loss": float(log_loss(y, p, labels=[0, 1]))}


def evaluation_report(y, raw, calibrated, seed=42, bootstrap_samples=200):
    y, raw, calibrated = map(np.asarray, (y, raw, calibrated))
    report = {"n": len(y), "defaults": int(y.sum()), "default_rate": float(y.mean()),
              "raw": metrics(y, raw), "calibrated": metrics(y, calibrated),
              "metric_definitions": {"average_precision": "Non-interpolated average precision; not trapezoidal PR area"},
              "bootstrap": {"seed": seed, "replicates": bootstrap_samples,
                            "scope": "Borrower bootstrap conditional on fitted models; excludes retraining uncertainty"}}
    rng = np.random.default_rng(seed)
    samples = {kind: [] for kind in ("raw", "calibrated")}
    for _ in range(bootstrap_samples):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        for kind, p in (("raw", raw), ("calibrated", calibrated)):
            samples[kind].append(metrics(y[idx], p[idx]))
    report["confidence_intervals_95"] = {
        kind: {key: np.quantile([s[key] for s in draws], [.025, .975]).tolist()
               for key in report[kind]} for kind, draws in samples.items() if draws}
    report["reliability"] = {}
    for kind, p in (("raw", raw), ("calibrated", calibrated)):
        actual, predicted = calibration_curve(y, p, n_bins=10, strategy="quantile")
        report["reliability"][kind] = {"mean_prediction": predicted.tolist(), "observed_rate": actual.tolist()}
    return report
