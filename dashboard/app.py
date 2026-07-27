"""
Credit Risk Analytics Engine — Streamlit Dashboard

Run from the project root:
    streamlit run dashboard/app.py

Requires artifacts/scored_portfolio.csv to already exist
(i.e. src/train.py -> src/calibration.py -> src/pipeline.py have all
been run once).
"""

import os
import sys

# Ensure project root is on the path regardless of how Streamlit boots
_here = os.path.abspath(__file__)                  # .../dashboard/app.py
_project_root = os.path.dirname(os.path.dirname(_here))  # .../CREDIT_RISK_ENGINE
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src import config
from src import scorecard
from src import stress_test

st.set_page_config(page_title="Credit Risk Analytics Engine", layout="wide")


@st.cache_data
def load_portfolio():
    return pd.read_csv(config.SCORED_PORTFOLIO_PATH)


st.title("Credit Risk Analytics Engine")
st.caption(
    "PD from calibrated LightGBM • 300-850 scorecard via WoE/log-odds scaling • "
    "Expected Loss stress-testing under simulated recession scenarios"
)

if not os.path.exists(config.SCORED_PORTFOLIO_PATH):
    st.error(
        f"No scored portfolio found at `{config.SCORED_PORTFOLIO_PATH}`.\n\n"
        "Run the pipeline first:\n"
        "1. `python -m src.train`\n"
        "2. `python -m src.calibration`\n"
        "3. `python -m src.pipeline`"
    )
    st.stop()

portfolio = load_portfolio()

tab_overview, tab_stress, tab_lookup = st.tabs(
    ["Portfolio Overview", "Stress Testing", "Individual Loan Lookup"]
)

# ------------------------------------------------------------------
# TAB 1 — Portfolio Overview
# ------------------------------------------------------------------
with tab_overview:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Portfolio size", f"{len(portfolio):,}")
    col2.metric("Mean PD (calibrated)", f"{portfolio['PD_CALIBRATED'].mean():.2%}")
    col3.metric("Mean Score", f"{portfolio['SCORE'].mean():.0f}")
    col4.metric("Baseline Portfolio EL", f"₹{portfolio['EL_baseline'].sum():,.0f}")

    c1, c2 = st.columns(2)
    with c1:
        fig = px.histogram(
            portfolio, x="SCORE", nbins=50, title="Credit Score Distribution (300-850)"
        )
        fig.add_vline(
            x=config.BASE_SCORE, line_dash="dash", annotation_text="Base score"
        )
        st.plotly_chart(fig, width="stretch")
    with c2:
        fig = px.histogram(
            portfolio, x="PD_CALIBRATED", nbins=50, title="Calibrated PD Distribution"
        )
        st.plotly_chart(fig, width="stretch")

    st.subheader("Risk Segment Breakdown")
    seg = (
        portfolio.groupby("SECURITY_CLASS")
        .agg(
            count=("SK_ID_CURR", "size"),
            mean_pd=("PD_CALIBRATED", "mean"),
            mean_lgd=("LGD", "mean"),
            total_ead=("EAD", "sum"),
            total_el_baseline=("EL_baseline", "sum"),
        )
        .reset_index()
    )
    st.dataframe(seg, width="stretch")

# ------------------------------------------------------------------
# TAB 2 — Stress Testing
# ------------------------------------------------------------------
with tab_stress:
    st.subheader("Expected Loss Shift Under Recession Scenarios")

    scenario_names = list(config.STRESS_SCENARIOS.keys())
    summary_rows = []
    for name in scenario_names:
        el_col = f"EL_{name}"
        summary_rows.append(
            {
                "scenario": name,
                "portfolio_EL": portfolio[el_col].sum(),
                "mean_PD": portfolio[f"PD_{name}"].mean(),
                "mean_LGD": portfolio[f"LGD_{name}"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)
    baseline_el = summary.loc[summary["scenario"] == "baseline", "portfolio_EL"].iloc[0]
    summary["EL_shift_pct"] = (
        100 * (summary["portfolio_EL"] - baseline_el) / baseline_el
    )

    c1, c2, c3 = st.columns(3)
    for col, name in zip([c1, c2, c3], scenario_names):
        row = summary[summary["scenario"] == name].iloc[0]
        col.metric(
            name.replace("_", " ").title(),
            f"₹{row['portfolio_EL']:,.0f}",
            f"{row['EL_shift_pct']:+.1f}%" if name != "baseline" else None,
        )

    fig = px.bar(
        summary,
        x="scenario",
        y="portfolio_EL",
        title="Portfolio Expected Loss by Scenario",
        text_auto=".2s",
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("---")
    st.subheader("Custom Stress Scenario")
    custom_shift = st.slider(
        "PD logit shift (higher = more severe recession)", 0.0, 2.0, 0.3, 0.05
    )
    custom_lgd_mult = st.slider(
        "LGD multiplier (collateral value stress)", 1.0, 2.0, 1.05, 0.01
    )

    try:
        pd_values = portfolio["PD_CALIBRATED"].values.astype(float)
        lgd_values = portfolio["LGD"].values.astype(float)
        ead_values = portfolio["EAD"].values.astype(float)

        # Guard against any NaN/Inf in the portfolio columns before computation
        if not (np.isfinite(pd_values).all() and np.isfinite(lgd_values).all() and np.isfinite(ead_values).all()):
            st.warning("Portfolio contains NaN or Inf values in PD/LGD/EAD columns — custom scenario skipped.")
        else:
            pd_custom = stress_test.stress_pd(pd_values, custom_shift)
            lgd_custom = np.clip(lgd_values * custom_lgd_mult, 0, 1)
            el_custom = stress_test.compute_expected_loss(pd_custom, lgd_custom, ead_values)

            # Replace any NaN/Inf that may arise from extreme shift values
            el_custom = np.nan_to_num(el_custom, nan=0.0, posinf=0.0, neginf=0.0)

            custom_el_total = float(el_custom.sum())
            custom_shift_pct = 100.0 * (custom_el_total - baseline_el) / baseline_el if baseline_el != 0 else 0.0

            c_m1, c_m2 = st.columns(2)
            c_m1.metric(
                "Custom Scenario Portfolio EL",
                f"₹{custom_el_total:,.0f}",
                f"{custom_shift_pct:+.1f}%",
            )
            c_m2.metric(
                "Mean Stressed PD",
                f"{pd_custom.mean():.2%}",
                f"{100*(pd_custom.mean() - pd_values.mean())/pd_values.mean():+.1f}% vs baseline" if pd_values.mean() != 0 else None,
            )
    except Exception as exc:
        st.error(
            f"Custom scenario calculation failed: {exc}\n\n"
            "Try adjusting the sliders to different values."
        )

# ------------------------------------------------------------------
# TAB 3 — Individual Loan Lookup
# ------------------------------------------------------------------
with tab_lookup:
    st.subheader("Look Up a Loan by SK_ID_CURR")
    loan_id = st.number_input(
        "SK_ID_CURR",
        min_value=int(portfolio["SK_ID_CURR"].min()),
        max_value=int(portfolio["SK_ID_CURR"].max()),
        step=1,
    )

    match = portfolio[portfolio["SK_ID_CURR"] == loan_id]
    if match.empty:
        st.warning("No loan found with that ID in the scored portfolio.")
    else:
        row = match.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Score", f"{row['SCORE']:.0f}")
        c2.metric("Calibrated PD", f"{row['PD_CALIBRATED']:.2%}")
        c3.metric("LGD", f"{row['LGD']:.0%}")
        c4.metric("EAD", f"₹{row['EAD']:,.0f}")

        st.write(f"**Security class:** {row['SECURITY_CLASS']}")

        scen_data = pd.DataFrame(
            {
                "scenario": scenario_names,
                "PD": [row[f"PD_{s}"] for s in scenario_names],
                "EL": [row[f"EL_{s}"] for s in scenario_names],
            }
        )
        st.dataframe(scen_data, width="stretch")
