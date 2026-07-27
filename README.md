# Credit Risk Analytics Engine

End-to-end production-style credit risk pipeline built on the Home Credit
Default Risk dataset (300,000+ relational loan records).

**Pipeline:** feature engineering across 6 relational tables → imbalanced
LightGBM default model (5-fold CV) → Isotonic Regression probability
calibration → WoE/log-odds 300-850 credit scorecard → LGD/EAD proxy
estimation → vectorized NumPy stress-testing of portfolio Expected Loss
under recession scenarios → Streamlit dashboard.

## Project structure

```
CREDIT_RISK_ENGINE/
├── DATA_SET/                   # place the 10 Home Credit CSVs here
├── artifacts/                  # generated — model, calibrator, scored data
├── src/
│   ├── config.py               # all paths, constants, and assumptions in one place
│   ├── feature_engineering.py  # builds the wide feature table
│   ├── train.py                # LightGBM training + diagnostics + artifact export
│   ├── calibration.py          # Isotonic Regression PD calibration
│   ├── scorecard.py            # PD -> 300-850 score, + feature-level WoE/IV
│   ├── lgd_ead.py              # LGD/EAD proxy assumptions (documented, not fitted)
│   ├── stress_test.py          # vectorized EL stress simulation
│   └── pipeline.py             # ties it all together -> scored_portfolio.csv
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Setup

```bash
cd CREDIT_RISK_ENGINE
pip install -r requirements.txt
```

Place the 10 Home Credit CSVs in `DATA_SET/`:
`application_train.csv`, `application_test.csv`, `bureau.csv`,
`bureau_balance.csv`, `previous_application.csv`, `POS_CASH_balance.csv`,
`installments_payments.csv`, `credit_card_balance.csv`,
`HomeCredit_columns_description.csv`, `sample_submission.csv`.

## Run order (each step depends on the previous one's artifacts)

```bash
# 1. Feature engineering + LightGBM training + diagnostics
python -m src.train

# 2. Isotonic Regression calibration (fits on OOF predictions)
python -m src.calibration

# 3. (optional) inspect scorecard math / feature-level WoE-IV
python -m src.scorecard

# 4. Full pipeline: builds the final scored portfolio with all
#    stress-test scenarios applied
python -m src.pipeline

# 5. Launch the dashboard
streamlit run dashboard/app.py
```

## Hugging Face Spaces deployment

This repository is now prepared for a Hugging Face Docker Space.

1. Create a new Hugging Face Space and choose the Docker SDK.
2. Connect the Space to this GitHub repository.
3. Keep the generated artifacts in `artifacts/` so the dashboard starts immediately.
4. Use the root `Dockerfile` to launch `app.py` on port `7860`.
5. Push updates to GitHub; the Space will redeploy from the main branch.

If you later want a source-only Space, we can switch to a startup pipeline that regenerates the artifacts inside Hugging Face instead of committing them.

## Key design notes

- **OOF, not in-sample, everywhere.** Calibration and scoring are fit on
  out-of-fold predictions from `train.py`'s 5-fold CV — never on
  training-set-in-sample predictions, which would be overconfident.
- **LGD/EAD are documented assumptions, not fitted models.** The Home
  Credit dataset has no recovery or exposure ground truth. `lgd_ead.py`
  uses Basel-style retail-portfolio benchmark values — change them in
  one place (`config.py` / `lgd_ead.py`), not scattered through the code.
- **PD is stressed via logit shift, not a flat probability shift** — this
  scales the stress by how much "room" a loan's probability has left,
  which is standard macro-stress-testing practice.
- **Everything in `stress_test.py` is vectorized NumPy** — no per-row
  Python loops — so the Streamlit custom-scenario slider recomputes the
  whole portfolio's Expected Loss in real time.
