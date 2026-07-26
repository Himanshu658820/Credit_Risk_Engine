"""
Credit Scorecard (300-850 scale) via log-odds scaling, plus feature-level
Weight of Evidence (WoE) / Information Value (IV) reporting.

Two things live here:

1. pd_to_score() — the actual scorecard: turns calibrated PD into a
   300-850 score using the standard log-odds scaling formula (same
   family as FICO):
       Score = Offset + Factor * ln(odds)
       Factor = PDO / ln(2)
       Offset = BaseScore - Factor * ln(BaseOdds)

2. compute_woe_iv() — feature-level WoE/IV binning, used for reporting
   *why* the model thinks what it thinks (which features are most
   discriminative) — standard credit-risk documentation practice, even
   though the score itself here comes from the calibrated model PD
   (approach "b" from our design discussion) rather than re-fit WoE
   logistic regression (approach "a").
"""

import numpy as np
import pandas as pd

from . import config


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def pd_to_score(calibrated_pd, base_score=None, base_odds=None, pdo=None,
                 score_min=None, score_max=None):
    """
    Convert calibrated PD (probability of default, "bad") into a score
    where HIGHER score = LOWER risk (standard credit-bureau convention).

    odds here = good:bad = (1 - PD) / PD
    """
    base_score = base_score or config.BASE_SCORE
    base_odds = base_odds or config.BASE_ODDS
    pdo = pdo or config.PDO
    score_min = score_min if score_min is not None else config.SCORE_MIN
    score_max = score_max if score_max is not None else config.SCORE_MAX

    calibrated_pd = np.clip(np.asarray(calibrated_pd, dtype=float), 1e-6, 1 - 1e-6)
    odds = (1 - calibrated_pd) / calibrated_pd

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    score = offset + factor * np.log(odds)
    return np.clip(score, score_min, score_max)


def score_to_pd(score, base_score=None, base_odds=None, pdo=None):
    """Inverse of pd_to_score — useful for sanity-checking the mapping."""
    base_score = base_score or config.BASE_SCORE
    base_odds = base_odds or config.BASE_ODDS
    pdo = pdo or config.PDO

    factor = pdo / np.log(2)
    offset = base_score - factor * np.log(base_odds)

    log_odds = (np.asarray(score, dtype=float) - offset) / factor
    odds = np.exp(log_odds)
    pd_ = 1 / (1 + odds)
    return pd_


def compute_woe_iv(df: pd.DataFrame, feature: str, target_col: str = "TARGET",
                    n_bins: int = None) -> pd.DataFrame:
    """
    Feature-level WoE/IV table for one numeric feature.
    WoE = ln( %good_in_bin / %bad_in_bin ), good = TARGET==0, bad = TARGET==1.
    IV per bin = (%good - %bad) * WoE ; total IV = sum over bins.
    """
    n_bins = n_bins or config.N_WOE_BINS
    tmp = df[[feature, target_col]].dropna().copy()
    tmp["bin"] = pd.qcut(tmp[feature], q=n_bins, duplicates="drop")

    grouped = tmp.groupby("bin", observed=True)[target_col].agg(["count", "sum"])
    grouped.columns = ["total", "bad"]
    grouped["good"] = grouped["total"] - grouped["bad"]

    total_good = grouped["good"].sum()
    total_bad = grouped["bad"].sum()

    # small epsilon avoids log(0) / div-by-0 in sparse bins
    eps = 0.5
    grouped["pct_good"] = (grouped["good"] + eps) / (total_good + eps * len(grouped))
    grouped["pct_bad"] = (grouped["bad"] + eps) / (total_bad + eps * len(grouped))
    grouped["woe"] = np.log(grouped["pct_good"] / grouped["pct_bad"])
    grouped["iv_bin"] = (grouped["pct_good"] - grouped["pct_bad"]) * grouped["woe"]

    grouped = grouped.reset_index()
    grouped["feature"] = feature
    grouped["iv_total"] = grouped["iv_bin"].sum()
    return grouped


def iv_strength_label(iv: float) -> str:
    """Standard credit-risk IV interpretation buckets."""
    if iv < 0.02:
        return "useless"
    elif iv < 0.1:
        return "weak"
    elif iv < 0.3:
        return "medium"
    elif iv < 0.5:
        return "strong"
    else:
        return "suspicious (check for leakage)"


def build_scorecard_report(df: pd.DataFrame, features: list, target_col: str = "TARGET") -> pd.DataFrame:
    """Runs compute_woe_iv over a list of features and returns a summary
    ranked by Information Value — the feature-level 'why' behind the score."""
    rows = []
    for feat in features:
        try:
            woe_df = compute_woe_iv(df, feat, target_col)
            iv = woe_df["iv_total"].iloc[0]
            rows.append({"feature": feat, "iv": iv, "strength": iv_strength_label(iv)})
        except Exception as e:
            rows.append({"feature": feat, "iv": np.nan, "strength": f"error: {e}"})
    report = pd.DataFrame(rows).sort_values("iv", ascending=False).reset_index(drop=True)
    return report


if __name__ == "__main__":
    oof_df = pd.read_csv(config.OOF_PREDICTIONS_PATH)

    from . import calibration
    calibrator = calibration.load_calibrator()
    oof_df["PD_CALIBRATED"] = calibration.apply_calibration(oof_df["PD_RAW"], calibrator)
    oof_df["SCORE"] = pd_to_score(oof_df["PD_CALIBRATED"])

    print("Score distribution:")
    print(oof_df["SCORE"].describe())

    # sanity check: inverse mapping should round-trip
    recovered_pd = score_to_pd(oof_df["SCORE"])
    max_err = np.max(np.abs(recovered_pd - oof_df["PD_CALIBRATED"]))
    print(f"\nRound-trip check (score -> pd vs original calibrated pd), max abs error: {max_err:.6f}")

    oof_df[["SK_ID_CURR", "TARGET", "PD_RAW", "PD_CALIBRATED", "SCORE"]].to_csv(
        config.SCORECARD_PATH, index=False
    )
    print(f"Saved scored portfolio: {config.SCORECARD_PATH}")
