"""
LGD (Loss Given Default) and EAD (Exposure at Default) proxy estimation.

IMPORTANT — READ THIS: Home Credit's public dataset has no recovery
amounts or realized-loss data, so there is no ground truth to fit LGD
against, and no drawn/undrawn balance history to fit a true EAD model
against either. Everything in this file is a documented, industry-
benchmark ASSUMPTION (loosely referenced from Basel retail-portfolio
norms), not an empirically-fitted estimate. Treat this file as the
single place these assumptions live — change them here, not downstream.
"""

import numpy as np
import pandas as pd

from . import config


def classify_security(row) -> str:
    """
    Proxy for whether a loan is "secured" (financing a specific good,
    e.g. POS/consumer loans where AMT_GOODS_PRICE ~ AMT_CREDIT) vs
    "unsecured" (pure cash loan, no linked asset) vs "revolving" (credit
    card). This is a coarse heuristic, not a legal determination of
    collateral status.
    """
    contract_type = row.get("_RAW_NAME_CONTRACT_TYPE", None)
    if contract_type == "Revolving loans":
        return "revolving"

    goods_price = row.get("_RAW_AMT_GOODS_PRICE", np.nan)
    credit = row.get("_RAW_AMT_CREDIT", np.nan)
    if pd.notna(goods_price) and pd.notna(credit) and credit > 0:
        # Loan closely tracks the price of a specific financed good
        # -> treat as (partially) secured.
        if 0.8 <= (goods_price / credit) <= 1.2:
            return "secured"
    return "unsecured"


def estimate_lgd(df: pd.DataFrame, lgd_multiplier: float = 1.0) -> pd.Series:
    """
    Returns per-loan LGD using the segment lookup in config.py.
    lgd_multiplier > 1.0 simulates recession stress (recovery values fall).
    """
    security = df.apply(classify_security, axis=1)
    lgd_map = {
        "secured": config.LGD_SECURED,
        "unsecured": config.LGD_UNSECURED,
        "revolving": config.LGD_REVOLVING,
    }
    lgd = security.map(lgd_map) * lgd_multiplier
    return lgd.clip(upper=1.0)


def estimate_ead(df: pd.DataFrame) -> pd.Series:
    """
    Returns per-loan EAD.
      - Non-revolving (cash/POS): EAD ~ AMT_CREDIT (full outstanding principal)
      - Revolving (credit card): EAD ~ drawn balance + CCF * undrawn limit
        Since application-level data doesn't carry live card balances,
        we approximate drawn balance as AMT_CREDIT (the same field Home
        Credit uses as the revolving line's credit amount) when a
        separate balance figure isn't available.
    """
    contract_type = df.get("_RAW_NAME_CONTRACT_TYPE", pd.Series("Cash loans", index=df.index))
    credit = df["_RAW_AMT_CREDIT"]

    is_revolving = (contract_type == "Revolving loans")
    ead = credit.copy()
    # For revolving loans, apply the CCF to the (assumed) undrawn portion.
    # Simplification: treat AMT_CREDIT as the limit and assume the loan is
    # currently ~50% drawn (no live balance field at application level).
    assumed_drawn_pct = 0.5
    revolving_ead = credit * assumed_drawn_pct + config.CCF_REVOLVING * credit * (1 - assumed_drawn_pct)
    ead = ead.where(~is_revolving, revolving_ead)
    return ead


def build_lgd_ead_table(df: pd.DataFrame, lgd_multiplier: float = 1.0) -> pd.DataFrame:
    """Convenience wrapper: returns SK_ID_CURR, LGD, EAD, SECURITY_CLASS."""
    out = pd.DataFrame({"SK_ID_CURR": df["SK_ID_CURR"]})
    out["SECURITY_CLASS"] = df.apply(classify_security, axis=1)
    out["LGD"] = estimate_lgd(df, lgd_multiplier=lgd_multiplier)
    out["EAD"] = estimate_ead(df)
    return out
