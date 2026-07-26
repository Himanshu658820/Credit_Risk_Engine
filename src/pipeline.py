"""
End-to-end pipeline: loads all fitted artifacts (model, calibrator) and
produces the final scored portfolio table used by the dashboard:

    SK_ID_CURR | PD_RAW | PD_CALIBRATED | SCORE | SECURITY_CLASS | LGD | EAD
    | EL_baseline | EL_mild_recession | EL_severe_recession | ...

Run directly (after train.py and calibration.py have been run once):
    python -m src.pipeline
"""

import pandas as pd

from . import config
from . import calibration
from . import scorecard
from . import lgd_ead
from . import stress_test


def build_scored_portfolio() -> pd.DataFrame:
    # 1. OOF predictions (already have PD_RAW, TARGET, and the _RAW_* columns
    #    train.py preserved for LGD/EAD classification)
    oof_df = pd.read_csv(config.OOF_PREDICTIONS_PATH)

    # 2. Calibrate
    calibrator = calibration.load_calibrator()
    oof_df["PD_CALIBRATED"] = calibration.apply_calibration(oof_df["PD_RAW"], calibrator)

    # 3. Score (300-850)
    oof_df["SCORE"] = scorecard.pd_to_score(oof_df["PD_CALIBRATED"])

    # 4. LGD / EAD (baseline, multiplier=1.0)
    lgd_ead_df = lgd_ead.build_lgd_ead_table(oof_df, lgd_multiplier=1.0)
    portfolio = oof_df.merge(lgd_ead_df, on="SK_ID_CURR", how="left")

    # 5. Stress test across all configured scenarios
    portfolio, summary = stress_test.run_all_scenarios(
        portfolio, pd_col="PD_CALIBRATED", lgd_col="LGD", ead_col="EAD"
    )

    return portfolio, summary


def score_new_loan(raw_features: dict) -> dict:
    """
    Scores a single new loan end-to-end (for dashboard "individual loan
    lookup"). raw_features must already be the fully feature-engineered
    row (same columns as training) — this does NOT re-run feature
    engineering, since that requires the full bureau/previous_application
    history joins. For a live system, feature_engineering would need a
    single-loan-compatible path; out of scope for this template.
    """
    import pickle
    import numpy as np

    with open(config.MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    X = pd.DataFrame([raw_features])[bundle["feature_names"]]
    fold_preds = [m.predict(X, num_iteration=m.best_iteration) for m in bundle["fold_models"]]
    pd_raw = float(np.mean(fold_preds))

    calibrator = calibration.load_calibrator()
    pd_calibrated = float(calibration.apply_calibration([pd_raw], calibrator)[0])
    score = float(scorecard.pd_to_score(pd_calibrated))

    return {"PD_RAW": pd_raw, "PD_CALIBRATED": pd_calibrated, "SCORE": score}


if __name__ == "__main__":
    portfolio, summary = build_scored_portfolio()
    portfolio.to_csv(config.SCORED_PORTFOLIO_PATH, index=False)
    print(f"Saved scored portfolio: {config.SCORED_PORTFOLIO_PATH}  shape={portfolio.shape}")

    print("\n=== Stress Test Summary ===")
    print(summary.to_string(index=False))
