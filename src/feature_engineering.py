"""
Feature engineering for the Home Credit dataset.
Builds one wide feature table joining application + 5 secondary tables.
"""

import os
import re
import gc
import numpy as np
import pandas as pd

from . import config


def reduce_mem_usage(df, name=""):
    """Downcast numeric dtypes to shrink memory footprint.

    Uses pd.api.types checks (not a raw `dtype != object` comparison) so it
    correctly skips non-numeric columns regardless of how pandas reports
    their dtype (object, string, category, pandas' newer 'str' dtype etc).
    Each column is wrapped in a try/except so one unexpected column can't
    crash the whole pipeline.
    """
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    skipped = []
    for col in df.columns:
        try:
            if pd.api.types.is_bool_dtype(df[col]):
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            col_type = df[col].dtype
            c_min, c_max = df[col].min(), df[col].max()
            if pd.isna(c_min) or pd.isna(c_max):
                continue

            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        except (TypeError, ValueError) as e:
            skipped.append((col, str(e)))
            continue

    end_mem = df.memory_usage(deep=True).sum() / 1024 ** 2
    print(f"[{name}] mem {start_mem:.1f}MB -> {end_mem:.1f}MB")
    if skipped:
        print(f"[{name}] skipped {len(skipped)} column(s) during downcast: "
              f"{[c for c, _ in skipped][:10]}{'...' if len(skipped) > 10 else ''}")
    return df


def one_hot(df, exclude_prefix="_RAW_"):
    """Dummy-encode any non-numeric, non-boolean column.

    Deliberately does NOT rely on `dtype == 'object'` — newer pandas
    (2.2+ with future.infer_string, pandas 3.x default) reports text
    columns as dtype 'str' / 'string[pyarrow]' instead of 'object',
    which a plain object check silently misses.

    Columns starting with `exclude_prefix` are left untouched — used to
    carry raw categorical values through to downstream modules (e.g.
    lgd_ead.py needs the raw NAME_CONTRACT_TYPE string, not its dummies).
    """
    cat_cols = [c for c in df.columns
                if not pd.api.types.is_numeric_dtype(df[c])
                and not pd.api.types.is_bool_dtype(df[c])
                and not str(c).startswith(exclude_prefix)]
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=True)
    return df, cat_cols


def sanitize_columns(df):
    """LightGBM rejects feature names containing JSON-special characters
    (: , " { } [ ]). Several Home Credit categorical values contain colons
    (e.g. ORGANIZATION_TYPE = 'Trade: type 3'), which leak into one-hot
    column names. Replace any non-alnum/underscore character with '_' and
    de-duplicate any resulting name collisions.
    """
    new_cols = []
    seen = {}
    for c in df.columns:
        clean = re.sub(r"[^0-9a-zA-Z_]+", "_", str(c))
        if clean in seen:
            seen[clean] += 1
            clean = f"{clean}_{seen[clean]}"
        else:
            seen[clean] = 0
        new_cols.append(clean)
    df.columns = new_cols
    return df


# ----------------------------------------------------------------------
# 1. APPLICATION TRAIN/TEST
# ----------------------------------------------------------------------
def process_application():
    train = pd.read_csv(os.path.join(config.DATA_DIR, "application_train.csv"))
    test = pd.read_csv(os.path.join(config.DATA_DIR, "application_test.csv"))
    print(f"application_train: {train.shape}, application_test: {test.shape}")

    train["is_train"] = 1
    test["is_train"] = 0
    test["TARGET"] = np.nan
    df = pd.concat([train, test], axis=0, ignore_index=True)

    # DAYS_EMPLOYED has a placeholder value of 365243 for "not employed"
    df["DAYS_EMPLOYED_ANOM"] = (df["DAYS_EMPLOYED"] == 365243).astype(int)
    df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)

    # Derived ratios
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"]
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    df["CREDIT_TERM"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"]
    df["DAYS_EMPLOYED_PERC"] = df["DAYS_EMPLOYED"] / df["DAYS_BIRTH"]
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"]
    df["INCOME_PER_FAM_MEMBER"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"]

    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_STD"] = df[ext_cols].std(axis=1)
    df["EXT_SOURCE_PROD"] = df[ext_cols].prod(axis=1)

    # Kept for LGD/EAD proxy logic downstream (collateral / contract type)
    df["_RAW_NAME_CONTRACT_TYPE"] = df["NAME_CONTRACT_TYPE"]
    df["_RAW_AMT_GOODS_PRICE"] = df["AMT_GOODS_PRICE"]
    df["_RAW_AMT_CREDIT"] = df["AMT_CREDIT"]

    df, _ = one_hot(df)
    df = reduce_mem_usage(df, "application")
    return df


# ----------------------------------------------------------------------
# 2. BUREAU + BUREAU_BALANCE
# ----------------------------------------------------------------------
def process_bureau():
    bureau = pd.read_csv(os.path.join(config.DATA_DIR, "bureau.csv"))
    bb = pd.read_csv(os.path.join(config.DATA_DIR, "bureau_balance.csv"))

    bb, _ = one_hot(bb)
    bb_agg = bb.groupby("SK_ID_BUREAU").agg(["mean", "sum", "max", "size"])
    bb_agg.columns = ["BB_" + "_".join(c).upper() for c in bb_agg.columns]
    bureau = bureau.join(bb_agg, on="SK_ID_BUREAU", how="left")
    del bb, bb_agg
    gc.collect()

    bureau, _ = one_hot(bureau)

    num_agg = {
        "DAYS_CREDIT": ["min", "max", "mean"],
        "DAYS_CREDIT_ENDDATE": ["min", "max", "mean"],
        "AMT_CREDIT_MAX_OVERDUE": ["mean"],
        "AMT_CREDIT_SUM": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_DEBT": ["max", "mean", "sum"],
        "AMT_CREDIT_SUM_OVERDUE": ["mean"],
        "CNT_CREDIT_PROLONG": ["sum"],
    }
    agg = bureau.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["BUREAU_" + "_".join(c).upper() for c in agg.columns]
    agg["BUREAU_COUNT"] = bureau.groupby("SK_ID_CURR").size()

    agg = reduce_mem_usage(agg.reset_index(), "bureau")
    return agg


# ----------------------------------------------------------------------
# 3. PREVIOUS_APPLICATION
# ----------------------------------------------------------------------
def process_previous_application():
    prev = pd.read_csv(os.path.join(config.DATA_DIR, "previous_application.csv"))
    prev["APP_CREDIT_PERC"] = prev["AMT_APPLICATION"] / prev["AMT_CREDIT"]

    prev, _ = one_hot(prev)

    num_agg = {
        "AMT_ANNUITY": ["mean", "max"],
        "AMT_APPLICATION": ["mean", "max"],
        "AMT_CREDIT": ["mean", "max"],
        "APP_CREDIT_PERC": ["mean", "max"],
        "AMT_DOWN_PAYMENT": ["mean"],
        "DAYS_DECISION": ["min", "max", "mean"],
        "CNT_PAYMENT": ["mean", "sum"],
    }
    agg = prev.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["PREV_" + "_".join(c).upper() for c in agg.columns]

    if "NAME_CONTRACT_STATUS_Approved" in prev.columns:
        agg["PREV_APPROVAL_RATE"] = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS_Approved"].mean()
    if "NAME_CONTRACT_STATUS_Refused" in prev.columns:
        agg["PREV_REFUSAL_RATE"] = prev.groupby("SK_ID_CURR")["NAME_CONTRACT_STATUS_Refused"].mean()
    agg["PREV_COUNT"] = prev.groupby("SK_ID_CURR").size()

    agg = reduce_mem_usage(agg.reset_index(), "previous_application")
    return agg


# ----------------------------------------------------------------------
# 4. POS_CASH_BALANCE
# ----------------------------------------------------------------------
def process_pos_cash():
    pos = pd.read_csv(os.path.join(config.DATA_DIR, "POS_CASH_balance.csv"))
    pos, _ = one_hot(pos)
    num_agg = {
        "MONTHS_BALANCE": ["max", "mean", "size"],
        "SK_DPD": ["max", "mean"],
        "SK_DPD_DEF": ["max", "mean"],
        "CNT_INSTALMENT_FUTURE": ["mean"],
    }
    agg = pos.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["POS_" + "_".join(c).upper() for c in agg.columns]
    agg["POS_COUNT"] = pos.groupby("SK_ID_CURR").size()
    agg = reduce_mem_usage(agg.reset_index(), "pos_cash")
    return agg


# ----------------------------------------------------------------------
# 5. INSTALLMENTS_PAYMENTS
# ----------------------------------------------------------------------
def process_installments():
    inst = pd.read_csv(os.path.join(config.DATA_DIR, "installments_payments.csv"))

    inst["PAYMENT_DELAY_DAYS"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["PAYMENT_UNDERPAY"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]
    inst["PAYMENT_PERC"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"]
    inst["LATE_FLAG"] = (inst["PAYMENT_DELAY_DAYS"] > 0).astype(int)

    num_agg = {
        "PAYMENT_DELAY_DAYS": ["mean", "max", "sum"],
        "PAYMENT_UNDERPAY": ["mean", "max", "sum"],
        "PAYMENT_PERC": ["mean", "min"],
        "LATE_FLAG": ["mean", "sum"],
        "AMT_INSTALMENT": ["sum", "mean"],
        "AMT_PAYMENT": ["sum", "mean"],
        "NUM_INSTALMENT_VERSION": ["nunique"],
    }
    agg = inst.groupby("SK_ID_CURR").agg(num_agg)
    agg.columns = ["INSTAL_" + "_".join(c).upper() for c in agg.columns]
    agg["INSTAL_COUNT"] = inst.groupby("SK_ID_CURR").size()
    agg = reduce_mem_usage(agg.reset_index(), "installments")
    return agg


# ----------------------------------------------------------------------
# 6. CREDIT_CARD_BALANCE
# ----------------------------------------------------------------------
def process_credit_card():
    cc = pd.read_csv(os.path.join(config.DATA_DIR, "credit_card_balance.csv"))
    cc, _ = one_hot(cc)
    cc["UTILIZATION"] = cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"]

    agg = cc.groupby("SK_ID_CURR").agg(["mean", "max", "sum"])
    agg.columns = ["CC_" + "_".join(c).upper() for c in agg.columns]
    agg["CC_COUNT"] = cc.groupby("SK_ID_CURR").size()
    agg = reduce_mem_usage(agg.reset_index(), "credit_card")
    return agg


# ----------------------------------------------------------------------
# BUILD FULL FEATURE TABLE
# ----------------------------------------------------------------------
def build_features():
    df = process_application()

    for builder in [process_bureau, process_previous_application,
                     process_pos_cash, process_installments, process_credit_card]:
        piece = builder()
        df = df.merge(piece, on="SK_ID_CURR", how="left")
        del piece
        gc.collect()

    leftover_obj_cols = [c for c in df.columns
                         if not pd.api.types.is_numeric_dtype(df[c])
                         and not pd.api.types.is_bool_dtype(df[c])
                         and c not in ("TARGET", "_RAW_NAME_CONTRACT_TYPE")]
    if leftover_obj_cols:
        print(f"WARNING: {len(leftover_obj_cols)} non-numeric column(s) "
              f"survived encoding, one-hot encoding them now: {leftover_obj_cols}")
        df = pd.get_dummies(df, columns=leftover_obj_cols, dummy_na=True)

    print(f"Final feature table: {df.shape}")
    return df
