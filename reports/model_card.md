# Credit Risk Engine model card

## Purpose and intended use

A portfolio research and interview demonstration of default prediction, calibration and model explanation using Home Credit Default Risk. Not intended for automated lending decisions, production underwriting or regulatory compliance.

## Model identity

Run: holdout-v1

Trained UTC: 2026-11-17T08:45:16.576470+00:00

Training source commit: 35a1c127539aebe0a1e4cc823866309a88b9c439

Training tree had local changes: True

Bundle SHA256: 6ae40ab361c4352e4242c988ca70902f0ae341e7fc6c6242c73217469678e21c

Feature schema SHA256: ab5406362733773647708367484c67a777da2260c64cbab6c3bb003b8203fd6d

The training source hashes and dataset file hashes are recorded in the run manifest. The commit alone does not identify local training changes.

## Data and evaluation design

Five LightGBM folds are fitted only within the development cohort, with early stopping on development validation rows. Their mean probability is fitted to an isotonic calibrator using a separate calibration cohort. The test cohort is used for final reporting, not threshold, model or risk-band selection.

| Cohort | Borrowers |
| --- | --- |
| development | 184,506 |
| calibration | 61,502 |
| test | 61,503 |

Split seed: 42. Test defaults: 4,965 (8.07%). This is a random borrower split within a dataset used previously for model development, not new external evidence. The target is the supplied binary TARGET; a standardized forward prediction horizon has not been established by this implementation.

## Held-out performance

| Metric | Raw ensemble | Calibrated ensemble |
| --- | --- | --- |
| roc_auc | 0.787761 | 0.787108 |
| average_precision | 0.287851 | 0.276842 |
| brier_score | 0.065638 | 0.065711 |
| log_loss | 0.236415 | 0.236801 |

Higher ROC-AUC and average precision are better; lower Brier score and log loss are better. Average precision is non-interpolated, not trapezoidal PR area.

Calibration did not improve Brier score or log loss on this test cohort. Both outputs are retained; no model is selected using this test comparison.

## Uncertainty

| Model | Metric | 95% lower | 95% upper |
| --- | --- | --- | --- |
| raw | roc_auc | 0.781426 | 0.794564 |
| raw | average_precision | 0.274561 | 0.300531 |
| raw | brier_score | 0.063895 | 0.067216 |
| raw | log_loss | 0.231196 | 0.241260 |
| calibrated | roc_auc | 0.780875 | 0.793850 |
| calibrated | average_precision | 0.264499 | 0.289029 |
| calibrated | brier_score | 0.063971 | 0.067270 |
| calibrated | log_loss | 0.231544 | 0.241900 |

Bootstrap replicates: 200. Borrower bootstrap conditional on fitted models; excludes retraining uncertainty

## Explanations and inputs

Native Tree SHAP explains one selected fold in raw log-odds. It does not decompose the ensemble probability or calibrated PD and does not establish causation. The dashboard uses 100 saved held-out examples; their importance chart is not a portfolio-wide estimate. Inputs must be fully engineered numeric features in the recorded schema. Missing history values are supported; infinite inputs are rejected.

## Loss assumptions

LGD proxies: secured 38%, unsecured 58%, revolving 65%. These are project assumptions, not fitted recovery estimates. The secured label uses a goods-price/credit ratio between 0.8 and 1.2; it is not evidence of legal collateral. Revolving EAD assumes 50% drawn plus a 75% conversion factor on the undrawn 50%; other EAD equals application credit. EAD stays constant under stress. Monetary amounts are in dataset units, with no currency conversion. Stress shifts are hypothetical and are not calibrated to macroeconomic outcomes.

The reporting layer uses the current project loss assumptions, separately from the trained PD bundle. Each borrower assessment records its scenario parameters. The 300–850 score is a custom log-odds mapping, not a bureau score.

## Limitations and governance

- Random borrower split, not temporal validation.
- Calibration is not guaranteed to improve every metric.
- Previously used dataset and model design: holdout is independent of this run, not a new external dataset.
- LGD/EAD remain assumptions; not evaluated by this PD report.

- Subgroup fairness, temporal drift and live outcome monitoring have not been validated.
- No approval policy or risk-band thresholds have been established.
- Artifact hashes detect mismatches; only trusted local pickle files may be loaded.
- Any material model change requires a new validation plan; repeatedly optimizing against this test cohort would compromise its independence.

## Reproduction

Run `python -m src.validated_training` with the Home Credit CSVs in `DATA_SET/`; then `streamlit run dashboard/app.py`. A new run directory is created. Run `python -m pytest tests -q` for numerical and contract tests.

| Training package | Version |
| --- | --- |
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| lightgbm | 4.7.0 |
| scikit-learn | 1.8.0 |
