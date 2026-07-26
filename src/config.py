"""
Central configuration for the Credit Risk Analytics Engine.
Every other module imports paths and constants from here.
"""

import os

# ----------------------------------------------------------------------
# PATHS  -- edit these two to match your machine
# ----------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "DATA_SET")

ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

OOF_PREDICTIONS_PATH = os.path.join(ARTIFACTS_DIR, "oof_predictions.csv")
MODEL_PATH = os.path.join(ARTIFACTS_DIR, "lgb_model.pkl")           # bundle: fold models + feature names
CALIBRATOR_PATH = os.path.join(ARTIFACTS_DIR, "calibrator.pkl")     # fitted IsotonicRegression
SCORECARD_PATH = os.path.join(ARTIFACTS_DIR, "scorecard_bins.csv")  # WoE/points mapping (feature-level)
SCORED_PORTFOLIO_PATH = os.path.join(ARTIFACTS_DIR, "scored_portfolio.csv")  # PD, LGD, EAD, EL, score per loan
SUBMISSION_PATH = os.path.join(ARTIFACTS_DIR, "submission.csv")

# ----------------------------------------------------------------------
# TRAINING CONSTANTS
# ----------------------------------------------------------------------
N_FOLDS = 5
RANDOM_SEED = 42
NUM_BOOST_ROUND = 10000
EARLY_STOPPING_ROUNDS = 200

LGB_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.02,
    "num_leaves": 34,
    "max_depth": 8,
    "min_child_samples": 70,
    "subsample": 0.87,
    "colsample_bytree": 0.9,
    "reg_alpha": 0.04,
    "reg_lambda": 0.073,
    "min_split_gain": 0.02,
    "verbose": -1,
    "seed": RANDOM_SEED,
}

# ----------------------------------------------------------------------
# SCORECARD CONSTANTS (300-850 scale, FICO-style)
# ----------------------------------------------------------------------
SCORE_MIN = 300
SCORE_MAX = 850

BASE_SCORE = 600       # score assigned at BASE_ODDS
BASE_ODDS = 50         # good:bad odds (50 good for every 1 bad) at BASE_SCORE
PDO = 20               # Points to Double the Odds
N_WOE_BINS = 10        # number of bins for feature-level WoE/IV reporting

# ----------------------------------------------------------------------
# LGD / EAD ASSUMPTIONS
# ----------------------------------------------------------------------
# NOTE: Home Credit's dataset has no recovery/exposure ground truth, so
# these are documented industry-benchmark ASSUMPTIONS, not empirically
# fitted values (loosely referenced from Basel retail-portfolio norms).
# Treat lgd_ead.py as the single place these get used/changed.

LGD_SECURED = 0.38        # goods-financed / POS-type loans (has a linked asset)
LGD_UNSECURED = 0.58      # pure cash loans (no linked asset)
LGD_REVOLVING = 0.65      # credit cards (typically highest LGD in retail)

CCF_REVOLVING = 0.75      # Credit Conversion Factor for undrawn revolving limit

# ----------------------------------------------------------------------
# STRESS TEST SCENARIOS
# ----------------------------------------------------------------------
# PD is stressed by shifting the log-odds (logit) of the calibrated PD.
# LGD is stressed multiplicatively (collateral/recovery value falls in
# a downturn). EAD is held constant (standard simplifying assumption).
STRESS_SCENARIOS = {
    "baseline": {"pd_logit_shift": 0.0, "lgd_multiplier": 1.00},
    "mild_recession": {"pd_logit_shift": 0.3, "lgd_multiplier": 1.05},
    "severe_recession": {"pd_logit_shift": 0.8, "lgd_multiplier": 1.15},
}
