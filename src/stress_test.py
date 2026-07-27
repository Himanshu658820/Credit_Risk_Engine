"""
Portfolio stress-testing: Expected Loss (EL = PD x LGD x EAD) under
simulated macroeconomic downturn scenarios.

Fully vectorized (NumPy arrays, no per-row loops) so this stays fast
enough to re-run live from a Streamlit slider.

Mechanism:
  - PD is stressed by shifting its LOGIT (log-odds), not the raw
    probability. This is standard practice — a flat additive shift on
    probability itself would push high-PD loans above 1.0 and treats a
    5% -> 10% move as equal in severity to a 90% -> 95% move, which a
    logit shift avoids (it scales the shift by how much "room" the
    probability has left).
  - LGD is stressed multiplicatively (recovery/collateral values fall
    in a downturn).
  - EAD is held constant (standard simplifying assumption for this kind
    of exercise — outstanding balances don't structurally change at the
    moment of default).
"""

import numpy as np
import pandas as pd

from . import config


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _sigmoid(x):
    with np.errstate(over="ignore"):
        result = 1 / (1 + np.exp(-x))
    # exp(-x) overflows to inf for very large negative x → result becomes 0 (correct)
    # exp(-x) underflows to 0 for very large positive x → result becomes 1 (correct)
    return np.nan_to_num(result, nan=0.5, posinf=1.0, neginf=0.0)


def stress_pd(pd_baseline, logit_shift: float):
    """Shift PD's log-odds by `logit_shift` and map back to probability."""
    pd_baseline = np.asarray(pd_baseline, dtype=float)
    return _sigmoid(_logit(pd_baseline) + logit_shift)


def compute_expected_loss(pd_arr, lgd_arr, ead_arr):
    """EL = PD * LGD * EAD, fully vectorized."""
    return np.asarray(pd_arr) * np.asarray(lgd_arr) * np.asarray(ead_arr)


def run_scenario(pd_baseline, lgd_baseline, ead, scenario_name: str):
    """
    Applies one named scenario from config.STRESS_SCENARIOS and returns
    per-loan stressed PD, LGD, EL.
    """
    if scenario_name not in config.STRESS_SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. "
                          f"Available: {list(config.STRESS_SCENARIOS.keys())}")
    params = config.STRESS_SCENARIOS[scenario_name]

    pd_stressed = stress_pd(pd_baseline, params["pd_logit_shift"])
    lgd_stressed = np.clip(np.asarray(lgd_baseline) * params["lgd_multiplier"], 0, 1)
    el_stressed = compute_expected_loss(pd_stressed, lgd_stressed, ead)

    return pd_stressed, lgd_stressed, el_stressed


def run_all_scenarios(portfolio: pd.DataFrame,
                       pd_col="PD_CALIBRATED", lgd_col="LGD", ead_col="EAD") -> pd.DataFrame:
    """
    portfolio must have columns [pd_col, lgd_col, ead_col].
    Returns (per-loan table with PD/LGD/EL for every configured scenario,
    portfolio-level summary DataFrame).
    """
    pd_baseline = portfolio[pd_col].values
    lgd_baseline = portfolio[lgd_col].values
    ead = portfolio[ead_col].values

    result = portfolio.copy()
    summary_rows = []

    for name in config.STRESS_SCENARIOS:
        pd_s, lgd_s, el_s = run_scenario(pd_baseline, lgd_baseline, ead, name)
        result[f"PD_{name}"] = pd_s
        result[f"LGD_{name}"] = lgd_s
        result[f"EL_{name}"] = el_s

        summary_rows.append({
            "scenario": name,
            "portfolio_EL": el_s.sum(),
            "mean_PD": pd_s.mean(),
            "mean_LGD": lgd_s.mean(),
            "total_EAD": ead.sum(),
        })

    summary = pd.DataFrame(summary_rows)
    baseline_el = summary.loc[summary["scenario"] == "baseline", "portfolio_EL"].iloc[0]
    summary["EL_shift_abs"] = summary["portfolio_EL"] - baseline_el
    summary["EL_shift_pct"] = 100 * summary["EL_shift_abs"] / baseline_el

    return result, summary


if __name__ == "__main__":
    # Smoke test with synthetic data (real run happens via pipeline.py,
    # which builds the actual scored portfolio from artifacts).
    np.random.seed(config.RANDOM_SEED)
    n = 10000
    demo_portfolio = pd.DataFrame({
        "SK_ID_CURR": np.arange(n),
        "PD_CALIBRATED": np.random.beta(2, 30, n),
        "LGD": np.random.choice([config.LGD_SECURED, config.LGD_UNSECURED, config.LGD_REVOLVING], n),
        "EAD": np.random.uniform(20000, 500000, n),
    })
    _, summary = run_all_scenarios(demo_portfolio)
    print("Stress test summary (synthetic demo data):")
    print(summary.to_string(index=False))
