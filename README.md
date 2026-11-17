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

### Independent evaluation workflow

Run `python -m src.validated_training` for the upgraded training workflow.
It splits labeled borrowers before feature construction into 60% development,
20% calibration, and 20% final test using seed 42. Five development folds select
their stopping iterations using only development validation rows. The ensemble's
mean probability is calibrated on the separate calibration cohort and evaluated
on the final test cohort. Category vocabularies for application and credit-card
features come only from development borrowers; unseen categories become missing.
Undefined feature ratios become missing values consistently.

Each run creates a new directory under `artifacts/validated/` containing a combined
model/calibrator bundle, split assignments, held-out predictions, metric report,
bootstrap intervals, reliability plot, sample engineered inputs, and a manifest
with source/data/artifact hashes and package versions. Existing dashboard artifacts
are not replaced. Do not load pickle artifacts from untrusted sources.

Use `--output <new-directory>` and `--threads 4` to control output and CPU use.
The output directory must not already exist. Run tests with
`python -m pytest tests -q` (install `pytest` separately).

The final test cohort must not be used to select models, thresholds, or risk bands.
This is a random split within a previously used dataset, not an external or temporal
validation. Average precision is reported explicitly rather than mislabeled as
trapezoidal PR-AUC. Bootstrap intervals condition on the fitted models and do not
include retraining uncertainty. Calibration can worsen some held-out metrics.

### Dashboard review and reports

The dashboard now opens on **Model validation and explanations**. Select a
completed run to view held-out metrics, reliability curves, bootstrap intervals,
and downloadable validation results. The **Why this risk?** tab explains one of
the 100 saved held-out feature rows, with a waterfall, observed inputs, and CSV
driver export. Native LightGBM Tree SHAP explains a selected fold's raw log-odds;
it does not decompose the ensemble probability or isotonic-calibrated PD.
The sample importance chart covers those 100 rows only. See the
[LightGBM prediction documentation](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)
for contribution semantics. No separate `shap` dependency is required.

The dashboard verifies artifact fingerprints and checks that saved inputs
reproduce evaluated predictions before displaying explanations. Only load trusted
local run folders: fingerprints detect mismatches, not malicious pickle files.
Validated artifacts are ignored by Git, so deployment requires copying a completed
run folder separately. Without one, the dashboard shows a setup message; the
**Legacy portfolio demo** remains available through the sidebar.

The **Why this risk?** tab also provides a downloadable Markdown borrower
assessment with a preview, verified PD and custom score, five explanation drivers,
and assumption-based LGD/EAD and stress losses. It derives exposures from the saved
engineered row; missing or ambiguous contract information makes loss estimates
unavailable. Reports record scenario parameters and use dataset monetary units.

The **Model card** tab provides a readable, downloadable account of model identity,
data splits, measured performance, uncertainty, assumptions and limitations.
`reports/model_card.md` records the reviewed `holdout-v1` run. Local borrower
examples stay under the ignored `artifacts/validated/` directory.

### Legacy demo workflow

The commands below rebuild the original artifacts. Their calibration checks use
the calibration fitting rows and are development diagnostics, not independent tests.

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

- **Legacy development diagnostics.** Calibration and scoring are fit on
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
